import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from multiprocessing import Pool, cpu_count
import numpy as np
from datetime import datetime
from collections import Counter
from A_dataset_fastbuild import load_and_preprocess_data
from B_dual_encoder_train import CBOW, load_cbow_embedding, process_batch_tokens, QryTower, DocTower, DualEncoder, collate_fn, load_query_tower, load_document_tower
from C_dataset_hardneg import load_and_preprocess_data_hard
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import pickle

def load_data(train_sample_size = None, val_sample_size = None, batch_size=300):

    # 1. Load training dataset of positives and hard negatives
    # train_dataset = load_and_preprocess_data_hard(filename = "hard_data_train", version="v1.1", split="train", dataset_sample_size=train_sample_size)
    # val_dataset = load_and_preprocess_data_hard(filename = "hard_data_val", version="v1.1", split="validation", dataset_sample_size=val_sample_size)
    with open('datasets/hard_data_train_8251.pkl', 'rb') as f:
        # Load the data from the file
        train_dataset = pickle.load(f)
    with open('datasets/hard_data_val_821.pkl', 'rb') as f:
        # Load the data from the file
        val_dataset = pickle.load(f)

    # 2. Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn = collate_fn, 
        num_workers = cpu_count()
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn = collate_fn, 
        num_workers = cpu_count()
    )

    return train_dataset, val_dataset, train_loader, val_loader

def finetune_DualEncoder_on_hardnegatives(train_loader, val_loader, model_type, batch_size=300, num_epochs=2, lr=0.001, margin = 0.3, run_name = None):

    # Initialize wandb
    if run_name is None:
        run_name = f"DualEncoder_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(project="dual-encoder-search", name=run_name)
    
    # Log hyperparameters
    config = {
        "model_type": model_type,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "margin": margin
    }
    wandb.config.update(config)

    #1. Load models and set up training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    query_tower, word2idx, idx2word = load_query_tower('models/query_tower_avgpool.pt')

    doc_tower, word2idx, idx2word = load_document_tower('models/document_tower_avgpool.pt')
    
    query_optimizer = torch.optim.Adam(query_tower.parameters(), lr = lr)
    doc_optimizer = torch.optim.Adam(doc_tower.parameters(), lr = lr)

    # Learning rate scheduler
    query_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(query_optimizer, 'min', patience=1, factor=0.5)
    doc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(doc_optimizer, 'min', patience=1, factor=0.5)

    dst_mrg = torch.tensor(margin).to(device)

    #2. Run training

    best_val_loss = float('inf')

    # Track metrics
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        query_tower.train()
        doc_tower.train()
        train_loss = 0
        batch_count = 0
        running_loss = 0
        all_train_difs = []
        
        for batch_idx, examples in enumerate(train_loader):
            
            # Forward pass
            query_optimizer.zero_grad()
            doc_optimizer.zero_grad()

            # Calculate cosine similarity
            # Process entire batches at once
            query_embeddings = query_tower(examples["query_tokens"])
            pos_doc_embeddings = doc_tower(examples["pos_passage_tokens"])
            neg_doc_embeddings = doc_tower(examples["neg_passage_tokens"])

            batch_dst_pos = F.cosine_similarity(query_embeddings, pos_doc_embeddings, dim=1)
            batch_dst_neg = F.cosine_similarity(query_embeddings, neg_doc_embeddings, dim=1)
            
            dst_dif = batch_dst_pos - batch_dst_neg
            all_train_difs.extend(dst_dif.detach().cpu().numpy())

            loss = torch.max(torch.tensor(0.0).to(device), dst_mrg - dst_dif).mean()

            # Backward pass
            loss.backward()
            query_optimizer.step()
            doc_optimizer.step()

            train_loss += loss.item()
            running_loss += loss.item()
            batch_count += 1

            if (batch_idx + 1) % 10 == 0:
                avg_running_loss = running_loss / 10
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"[Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_loader)} ({progress:.1f}%) - Loss: {avg_running_loss:.6f}")
                running_loss = 0

                # Log batch metrics to wandb
                wandb.log({
                    "batch_loss": avg_running_loss,
                    "batch": batch_idx + 1 + epoch * len(train_loader),
                    "learning_rate": query_optimizer.param_groups[0]['lr']
                })

        avg_train_loss = train_loss / batch_count
        train_losses.append(avg_train_loss)

        # Calculate training metrics
        train_predictions = [1 if d > 0 else 0 for d in all_train_difs]
        train_labels = [1] * len(train_predictions)  # We want positive pairs to have higher similarity
        train_accuracy = sum(train_predictions) / len(train_predictions)
       
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # 3. Run validation
        query_tower.eval()
        doc_tower.eval()
        val_loss = 0
        val_batch_count = 0
        all_val_difs = []

        with torch.no_grad():
            for examples in val_loader:

                # Calculate cosine similarity

                # Process entire batches at once
                query_embeddings = query_tower(examples["query_tokens"])
                pos_doc_embeddings = doc_tower(examples["pos_passage_tokens"])
                neg_doc_embeddings = doc_tower(examples["neg_passage_tokens"])

                batch_dst_pos = F.cosine_similarity(query_embeddings, pos_doc_embeddings, dim=1)
                batch_dst_neg = F.cosine_similarity(query_embeddings, neg_doc_embeddings, dim=1)
                
                dst_dif = batch_dst_pos - batch_dst_neg

                all_val_difs.extend(dst_dif.detach().cpu().numpy())

                loss = torch.max(torch.tensor(0.0).to(device), dst_mrg - dst_dif).mean()

                val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        
        # Calculate validation metrics
        val_predictions = [1 if d > 0 else 0 for d in all_val_difs]
        val_labels = [1] * len(val_predictions)  # We want positive pairs to have higher similarity
        val_accuracy = sum(val_predictions) / len(val_predictions)
        
        # Calculate precision, recall, and F1 score
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_predictions, average='binary', zero_division=0
        )
        
        # Calculate mean and median similarity differences
        mean_val_dif = np.mean(all_val_difs)
        median_val_dif = np.median(all_val_difs)
        
        # Calculate percentage of pairs where positive similarity > negative similarity
        pct_positive_higher = (np.array(all_val_difs) > 0).mean() * 100
        
        # Store current metrics
        current_metrics = {
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall, 
            "val_f1": val_f1,
            "mean_sim_diff": mean_val_dif,
            "median_sim_diff": median_val_dif,
            "pct_positive_higher": pct_positive_higher
        }
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Metrics:")
        print(f"  Accuracy: {val_accuracy:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        print(f"  F1 Score: {val_f1:.4f}")
        print(f"  Mean Similarity Difference: {mean_val_dif:.4f}")
        print(f"  Median Similarity Difference: {median_val_dif:.4f}")
        print(f"  % Positive > Negative: {pct_positive_higher:.2f}%")
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "mean_sim_diff": mean_val_dif,
            "median_sim_diff": median_val_dif,
            "pct_positive_higher": pct_positive_higher,
        })
        
        # Update learning rate scheduler
        query_scheduler.step(avg_val_loss)
        doc_scheduler.step(avg_val_loss)

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_metrics = current_metrics
            print(f"New best model with validation loss: {best_val_loss:.4f}")

            save_tower_finetuned(query_tower, word2idx, idx2word, filename = 'qrytoweravg_hard') 
            save_tower_finetuned(doc_tower, word2idx, idx2word, filename = 'doctoweravg_hard')
            
            # Save best metrics to wandb
            for key, value in best_val_metrics.items():
                wandb.summary[f"best_{key}"] = value
            wandb.summary["best_val_loss"] = best_val_loss
    
    # Log final best values
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Best validation metrics:")
    for key, value in best_val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Close wandb run
    wandb.finish()
    
    return train_losses, val_losses, best_val_metrics, query_tower, doc_tower


def save_tower_finetuned(model, word2idx, idx2word, filename):
    """Save model with vocabulary mappings"""
    print(f"Saving model to {filename}.pt...")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'word2idx': word2idx,
        'idx2word': idx2word,
    }, f"{filename}.pt")
    print(f"Model saved to {filename}.pt")

def maintrain(train_sample_size=100, val_sample_size = 20, batch_size=300, num_epochs=5, lr=0.001, model_type="AvgPool", run_name=None):
    #1 Load CBOW embeddings
    embedding, word2idx, idx2word, embedding_dim = load_cbow_embedding()
    #2 Load dataset
    train_dataset, val_dataset, train_loader, val_loader = load_data(train_sample_size, val_sample_size, batch_size)
    # Log dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    #3 Train and save model
    train_losses, val_losses, best_metrics, query_tower, doc_tower = finetune_DualEncoder_on_hardnegatives(
        train_loader, 
        val_loader, 
        model_type=model_type, # check model_type of QryTower and DocTowxer (Rnn or AvgPool)
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        lr=lr,
        run_name=run_name
    ) 

    # Display final results
    print("\nTraining Summary:")
    print(f"Model type: {model_type}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation metrics:")
    for key, value in best_metrics.items():
        print(f"  {key}: {value:.4f}")

    return train_losses, val_losses, best_metrics

if __name__ == "__main__":
    maintrain()


        





