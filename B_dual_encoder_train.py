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

class QryTower(nn.Module):
    def __init__(self, embedding_layer, embedding_dim, word2idx, hidden_size = 256, device = 'cpu'):
        super(QryTower, self).__init__()
        self.embedding = embedding_layer
        self.embedding_dim = embedding_dim
        self.word2idx = word2idx
        self.device = device
        # #Option 1 : average pooling and MLP + batch norm + drop out
        # self.fc1 = nn.Linear(self.embedding_dim, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.dropout = nn.Dropout(0.2)

        #Option 2 : RNN + final FF layer + drop out
        self.rnn = nn.RNN(input_size=embedding_dim, 
                          hidden_size=hidden_size, 
                          batch_first=True,
                          num_layers=2,
                          dropout=0.2)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, batch_query_tokens):
        # #Option 1 : average pooling and MLP + batch norm + drop out
        # # Process an entire batch at once
        # batch_size = len(batch_query_tokens)
        # batch_embeddings = []
        # for query_tokens in batch_query_tokens:
        #     # Convert tokens to indices
        #     query_token_idxs = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in query_tokens]
        #     query_token_idxs = torch.tensor(query_token_idxs, dtype=torch.long, device=self.device)
        #     query_embeddings = self.embedding(query_token_idxs)
        #     avg_query_embedding = torch.mean(query_embeddings, dim=0)  # Average embeddings
        #     batch_embeddings.append(avg_query_embedding)
        # # Stack embeddings to create a batch
        # batch_embeddings = torch.stack(batch_embeddings)

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
        
        # #Option 1 : average pooling and MLP + batch norm + drop out
        # x = self.fc1(batch_embeddings)
        # x = self.bn1(x)  # Apply batch normalization
        # x = F.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = self.bn2(x)  # Apply batch normalization
        # x = F.relu(x)
        # x = self.dropout(x)
        # QryEmbeddings = self.fc3(x)  # Shape: [batch_size, 1]

        #Option 2 : RNN + final FF layer + drop out
        QryEmbeddings = self.fc(self.dropout(batch_embeddings))
        
        return QryEmbeddings

class DocTower(nn.Module):
    def __init__(self, embedding_layer, embedding_dim, word2idx, hidden_size = 256, device='cpu'):
        super(DocTower, self).__init__()
        self.embedding = embedding_layer
        self.embedding_dim = embedding_dim
        self.word2idx = word2idx
        self.device = device

        # #Option 1 : average pooling and MLP + batch norm + drop out
        # self.fc1 = nn.Linear(self.embedding_dim, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)  # Add batch normalization
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.dropout = nn.Dropout(0.2)

        #Option 2 : RNN + final FF layer + drop out
        self.rnn = nn.RNN(input_size=embedding_dim, 
                          hidden_size=hidden_size, 
                          batch_first=True,
                          num_layers=2,
                          dropout=0.2)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, batch_passage_tokens):
        # # #Option 1 : average pooling and MLP + batch norm + drop out
        # # Process an entire batch at once
        # batch_size = len(batch_passage_tokens)
        # batch_embeddings = []
        # for passage_tokens in batch_passage_tokens:
        #     # Convert tokens to indices
        #     passage_token_idxs = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in passage_tokens]
        #     passage_token_idxs = torch.tensor(passage_token_idxs, dtype=torch.long, device=self.device)
        #     passage_embeddings = self.embedding(passage_token_idxs)
        #     avg_passage_embedding = torch.mean(passage_embeddings, dim=0)  # Average embeddings
        #     batch_embeddings.append(avg_passage_embedding)
        # # Stack embeddings to create a batch
        # batch_embeddings = torch.stack(batch_embeddings)

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
        
        # #Option 1 : average pooling and MLP + batch norm + drop out
        # x = self.fc1(batch_embeddings)
        # x = self.bn1(x)  # Apply batch normalization
        # x = F.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = self.bn2(x)  # Apply batch normalization
        # x = F.relu(x)
        # x = self.dropout(x)
        # DocEmbeddings = self.fc3(x)  # Shape: [batch_size, 1]

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

def collate_fn(batch):
    batch_dict = {
        "query_tokens": [item["query_tokens"] for item in batch],
        "pos_passage_tokens": [item["pos_passage_tokens"] for item in batch],
        "neg_passage_tokens": [item["neg_passage_tokens"] for item in batch]
    }
    return batch_dict

def load_data(train_sample_size = None, val_sample_size = None, batch_size=300):

    #1. Load training dataset
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

def train_DualEncoder(train_loader, val_loader, batch_size=300, num_epochs=2, lr=0.001):

        #1. Load models and set up training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model = DualEncoder()
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

        dst_mrg = torch.tensor(0.3).to(device)

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
            
            for batch_idx, examples in enumerate(train_loader):
                
                # Forward pass
                optimizer.zero_grad()

                dst_dif = model(examples)

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

            avg_train_loss = train_loss / batch_count
            train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

            # 3. Run validation
            model.eval()
            val_loss = 0
            val_batch_count = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for examples in val_loader:

                    dst_dif = model(examples)

                    loss = torch.max(torch.tensor(0.0).to(device), dst_mrg - dst_dif).mean()

                    val_loss += loss.item()
                    val_batch_count += 1

            avg_val_loss = val_loss / val_batch_count

            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"Saved query and doc tower models separately with improved validation loss: {best_val_loss:.4f}")
                # Save towers separately
                save_separate_towers(model)
        
        print("Training complete!")

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

def maintrain(train_sample_size=None, val_sample_size=None, batch_size=300, num_epochs=5, lr=0.001):
    #1 Load CBOW embeddings
    embedding, word2idx, idx2word, embedding_dim = load_cbow_embedding()
    #2 Load dataset
    train_dataset, val_dataset, train_loader, val_loader = load_data(train_sample_size, val_sample_size)
    #3 Train and save model
    train_DualEncoder(train_loader, val_loader, batch_size, num_epochs, lr)



if __name__ == "__main__":
    maintrain()


        





