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
from C_dataset_hardneg import load_and_preprocess_data_hard
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support


class CBOW(torch.nn.Module): 
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        # Average embeddings from context words
        embedded = self.embedding(inputs)
        embedded = torch.mean(embedded, dim=1)
        # Project to vocabulary space
        output = self.linear(embedded)
        # Apply log softmax for numerical stability
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

def load_cbow_embedding():
    # Download model from HuggingFace
    model_file = hf_hub_download(
        repo_id="Apples96/cbow_model", 
        filename="output/cbow_model_full.pt"
    )
    
    # Load model
    checkpoint = torch.load(model_file)
    
    # Extract components
    word2idx = checkpoint['token_to_index']
    idx2word = checkpoint['index_to_token']
    embedding_dim = checkpoint['embedding_dim']
    vocab_size = checkpoint['vocab_size']
    
    # Create model
    model = CBOW(vocab_size, embedding_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get embedding layer and freeze it
    embedding = model.embedding
    for param in embedding.parameters():
        param.requires_grad = False
    
    return embedding, word2idx, idx2word, embedding_dim


class QryTower(nn.Module):
    def __init__(self, embedding_layer, embedding_dim, word2idx, model_type = "AvgPool", hidden_size = 256, device = 'cpu'):
        super(QryTower, self).__init__()
        self.embedding = embedding_layer
        self.embedding_dim = embedding_dim
        self.word2idx = word2idx
        self.device = device
        self.model_type = model_type

        #Option 1 : average pooling and MLP + batch norm + drop out
        # Choose the tower types based on model_type parameter
        if model_type == "AvgPool":
            self.fc1 = nn.Linear(self.embedding_dim, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(0.2)
        elif model_type == "RNN":
            self.rnn = nn.RNN(input_size=embedding_dim, 
                          hidden_size=hidden_size, 
                          batch_first=True,
                          num_layers=1,
                          dropout=0.0)
            self.fc = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(0.2)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Expected 'AvgPool' or 'RNN'.")
    
    def forward(self, batch_query_tokens):
        #Option 1 : average pooling and MLP + batch norm + drop out
        if self.model_type == "AvgPool":
        # Process an entire batch at once
            batch_size = len(batch_query_tokens)
            batch_embeddings = []
            for query_tokens in batch_query_tokens:
                # Convert tokens to indices
                query_token_idxs = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in query_tokens]
                query_token_idxs = torch.tensor(query_token_idxs, dtype=torch.long, device=self.device)
                query_embeddings = self.embedding(query_token_idxs)
                avg_query_embedding = torch.mean(query_embeddings, dim=0)  # Average embeddings
                batch_embeddings.append(avg_query_embedding)
            # Stack embeddings to create a batch
            batch_embeddings = torch.stack(batch_embeddings)
            
            #Option 1 : average pooling and MLP + batch norm + drop out
            x = self.fc1(batch_embeddings)
            x = self.bn1(x)  # Apply batch normalization
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.bn2(x)  # Apply batch normalization
            x = F.relu(x)
            x = self.dropout(x)
            QryEmbeddings = self.fc3(x)  # Shape: [batch_size, 1]
        elif self.model_type == "RNN":
            #Option 2 : RNN + final FF layer + drop out
            # Optimization: Process all sequences in a single batch operation
            padded_indices, lengths = process_batch_tokens(batch_query_tokens, self.word2idx, device=self.device)
            # Get embeddings for all sequences at once
            batch_embeddings = self.embedding(padded_indices)  # [batch_size, max_seq_len, embedding_dim]
            # Pack padded sequences for efficient RNN processing
            packed_embeddings = nn.utils.rnn.pack_padded_sequence(
                batch_embeddings, 
                lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            _, hidden = self.rnn(packed_embeddings) # Process through RNN (much more efficient on packed sequences)
            
            # Get the final hidden state from the last layer
            batch_embeddings = hidden[-1]  # [batch_size, hidden_size]

            #Option 2 : RNN + final FF layer + drop out
            QryEmbeddings = self.fc(self.dropout(batch_embeddings))
        
        return QryEmbeddings

class DocTower(nn.Module):
    def __init__(self, embedding_layer, embedding_dim, word2idx, model_type = "AvgPool", hidden_size = 256, device='cpu'):
        super(DocTower, self).__init__()
        self.embedding = embedding_layer
        self.embedding_dim = embedding_dim
        self.word2idx = word2idx
        self.device = device
        self.model_type = model_type

        if model_type == "AvgPool":
            #Option 1 : average pooling and MLP + batch norm + drop out
            self.fc1 = nn.Linear(self.embedding_dim, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(0.2)
        elif model_type == "RNN":
            #Option 2 : RNN + final FF layer + drop out
            self.rnn = nn.RNN(input_size=embedding_dim, 
                            hidden_size=hidden_size, 
                            batch_first=True,
                            num_layers=1,
                            dropout=0.0)
            self.fc = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(0.2)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Expected 'AvgPool' or 'RNN'.")
    
    def forward(self, batch_passage_tokens):
        # #Option 1 : average pooling and MLP + batch norm + drop out
        if self.model_type == "AvgPool":
            # Process an entire batch at once
            batch_size = len(batch_passage_tokens)
            batch_embeddings = []
            for passage_tokens in batch_passage_tokens:
                # Convert tokens to indices
                passage_token_idxs = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in passage_tokens]
                passage_token_idxs = torch.tensor(passage_token_idxs, dtype=torch.long, device=self.device)
                passage_embeddings = self.embedding(passage_token_idxs)
                avg_passage_embedding = torch.mean(passage_embeddings, dim=0)  # Average embeddings
                batch_embeddings.append(avg_passage_embedding)
            # Stack embeddings to create a batch
            batch_embeddings = torch.stack(batch_embeddings)
            
            #Option 1 : average pooling and MLP + batch norm + drop out
            x = self.fc1(batch_embeddings)
            x = self.bn1(x)  # Apply batch normalization
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.bn2(x)  # Apply batch normalization
            x = F.relu(x)
            x = self.dropout(x)
            DocEmbeddings = self.fc3(x)  # Shape: [batch_size, 1]
        elif self.model_type == "RNN":
            #Option 2 : RNN + final FF layer + drop out
            # Optimization: Process all sequences in a single batch operation
            padded_indices, lengths = process_batch_tokens(batch_passage_tokens, self.word2idx, device=self.device)
            # Get embeddings for all sequences at once
            batch_embeddings = self.embedding(padded_indices)  # [batch_size, max_seq_len, embedding_dim]
            # Pack padded sequences for efficient RNN processing
            packed_embeddings = nn.utils.rnn.pack_padded_sequence(
                batch_embeddings, 
                lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            # Process through RNN (much more efficient on packed sequences)
            _, hidden = self.rnn(packed_embeddings)
            
            # Get the final hidden state from the last layer
            batch_embeddings = hidden[-1]  # [batch_size, hidden_size]

            # Option 2 : RNN + final FF layer + drop out
            DocEmbeddings = self.fc(self.dropout(batch_embeddings))

        return DocEmbeddings


class DualEncoder(nn.Module):
    def __init__(self):
        super(DualEncoder, self).__init__()
        # Load CBOW embedding
        embedding, word2idx, idx2word, embedding_dim = load_cbow_embedding()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize the query and document towers
        self.query_tower = QryTower(embedding, embedding_dim, word2idx, device=self.device)
        self.doc_tower = DocTower(embedding, embedding_dim, word2idx, device=self.device)
        # Store word2idx and idx2word
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def forward(self, examples):
        # Process entire batches at once
        query_embeddings = self.query_tower(examples["query_tokens"])
        pos_doc_embeddings = self.doc_tower(examples["pos_passage_tokens"])
        neg_doc_embeddings = self.doc_tower(examples["neg_passage_tokens"])
        
        # Calculate cosine similarity
        batch_dst_pos = F.cosine_similarity(query_embeddings, pos_doc_embeddings, dim=1)
        batch_dst_neg = F.cosine_similarity(query_embeddings, neg_doc_embeddings, dim=1)
        
        dst_dif = batch_dst_pos - batch_dst_neg
        return dst_dif

def load_data(train_sample_size = None, val_sample_size = None, batch_size=300):

    # Option 1 : load dataset of positives and random negatives
    train_dataset = load_and_preprocess_data(version="v1.1", split="train", dataset_sample_size=train_sample_size)
    val_dataset = load_and_preprocess_data(version="v1.1", split="validation", dataset_sample_size=val_sample_size)

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

# Add a pre-processing function to convert tokens to indices in batches
def process_batch_tokens(batch_tokens, word2idx, max_len=None, device='cpu'):
    """Convert a batch of token sequences to padded tensor of indices"""
    # Get UNK token index
    unk_idx = word2idx.get('<UNK>', 0)
    
    # Convert tokens to indices
    batch_tokenindices = [[word2idx.get(token, unk_idx) for token in tokens] for tokens in batch_tokens]
    
    # Get lengths before padding
    lengths = torch.tensor([len(indices) for indices in batch_tokenindices], device=device)
    
    # Determine padding length if not specified
    if max_len is None:
        max_len = max(len(indices) for indices in batch_tokenindices)
    
    # Pad sequences
    padded_tokenindices = [indices + [0] * (max_len - len(indices)) for indices in batch_tokenindices]
    
    # Convert to tensor
    padded_tokenindices = torch.tensor(padded_tokenindices, dtype=torch.long, device=device)
    
    return padded_tokenindices, lengths

def collate_fn(batch):
    batch_dict = {
        "query_tokens": [item["query_tokens"] for item in batch],
        "pos_passage_tokens": [item["pos_passage_tokens"] for item in batch],
        "neg_passage_tokens": [item["neg_passage_tokens"] for item in batch]
    }
    return batch_dict

def train_DualEncoder(train_loader, val_loader, model_type, batch_size=300, num_epochs=2, lr=0.001, margin = 0.3, run_name = None):

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

    model = DualEncoder()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)

    dst_mrg = torch.tensor(margin).to(device)

    #2. Run training

    best_val_loss = float('inf')

    # Track metrics
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        batch_count = 0
        running_loss = 0
        all_train_difs = []
        
        for batch_idx, examples in enumerate(train_loader):
            
            # Forward pass
            optimizer.zero_grad()

            dst_dif = model(examples)

            all_train_difs.extend(dst_dif.detach().cpu().numpy())

            loss = torch.max(torch.tensor(0.0).to(device), dst_mrg - dst_dif).mean()

            # Backward pass
            loss.backward()
            optimizer.step()

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
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

        avg_train_loss = train_loss / batch_count
        train_losses.append(avg_train_loss)

        # Calculate training metrics
        train_predictions = [1 if d > 0 else 0 for d in all_train_difs]
        train_labels = [1] * len(train_predictions)  # We want positive pairs to have higher similarity
        train_accuracy = sum(train_predictions) / len(train_predictions)
       
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # 3. Run validation
        model.eval()
        val_loss = 0
        val_batch_count = 0
        all_val_difs = []

        with torch.no_grad():
            for examples in val_loader:

                dst_dif = model(examples)
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
        scheduler.step(avg_val_loss)

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_metrics = current_metrics
            print(f"New best model with validation loss: {best_val_loss:.4f}")
            # Save towers separately
            save_separate_towers(model)
            
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
    
    return train_losses, val_losses, best_val_metrics


def save_separate_towers(model):
    """Save query tower and document tower as separate models"""
    print("Saving query tower and document tower separately...")
    
    # Save query tower
    torch.save({
        'model_state_dict': model.query_tower.state_dict(),
        'word2idx': model.word2idx,
        'idx2word': model.idx2word,
    }, f"query_tower.pt")
    print(f"Query tower saved to query_tower.pt")
    
    # Save document tower
    torch.save({
        'model_state_dict': model.doc_tower.state_dict(),
        'word2idx': model.word2idx,
        'idx2word': model.idx2word,
    }, f"document_tower.pt")
    print(f"Document tower saved to document_tower.pt")

def load_query_tower(filepath="query_tower_avgpool.pt", device='cpu'):
    """Load the query tower model"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Get embedding layer and word2idx
    embedding, word2idx, idx2word, embedding_dim = load_cbow_embedding()
    
    # Create query tower
    query_tower = QryTower(embedding, embedding_dim, word2idx, device=device)
    query_tower.load_state_dict(checkpoint['model_state_dict'])
    query_tower.to(device)
    query_tower.eval()
    
    return query_tower, checkpoint['word2idx'], checkpoint['idx2word']

def load_document_tower(filepath="document_tower.pt", device='cpu'):
    """Load the document tower model"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Get embedding layer and word2idx
    embedding, word2idx, idx2word, embedding_dim = load_cbow_embedding()
    
    # Create document tower
    doc_tower = DocTower(embedding, embedding_dim, word2idx, device=device)
    doc_tower.load_state_dict(checkpoint['model_state_dict'])
    doc_tower.to(device)
    doc_tower.eval()
    
    return doc_tower, checkpoint['word2idx'], checkpoint['idx2word']

def maintrain(train_sample_size=100000, val_sample_size=1000, batch_size=300, num_epochs=2, lr=0.001, model_type="AvgPool", run_name=None):
    #1 Load CBOW embeddings
    embedding, word2idx, idx2word, embedding_dim = load_cbow_embedding()
    #2 Load dataset
    train_dataset, val_dataset, train_loader, val_loader = load_data(train_sample_size, val_sample_size, batch_size)
    # Log dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    #3 Train and save model
    train_losses, val_losses, best_metrics = train_DualEncoder(
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


        





