import torch
from datasets import load_dataset
import weaviate
from weaviate.classes.init import Auth
import os
from A_dataset_fastbuild import tokenize_text
from B_dual_encoder_train import DocTower, CBOW, load_cbow_embedding
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load document_tower model
def load_doctower():
    checkpoint = torch.load("document_tower_avgpool.pt")
    embedding, word2idx, idx2word, embedding_dim = load_cbow_embedding()
    doc_tower = DocTower(embedding, embedding_dim, word2idx)
    doc_tower.load_state_dict(checkpoint['model_state_dict'])
    doc_tower.eval()
    return doc_tower
    print(f"DocTower model is loaded")

# Load all the documents in each of training, velidation and test sets
def load_all_passages(max_passages=None):
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    all_passages = []
    count = 0

    # You need to iterate over each split separately
    for split_name in ['train', 'validation', 'test']:
        split_dataset = dataset[split_name]
        
        for item in split_dataset:
            passage_texts = item['passages']['passage_text']
                
            # Add each passage to our collection
            for passage in passage_texts:
                all_passages.append(passage)
                count += 1
                if max_passages is not None and count >= max_passages:
                    print(f"Loaded {len(all_passages)} passages (limited to {max_passages})")
                    return all_passages

    print(f"Total passages collected: {len(all_passages)}")
    return all_passages
    

# Run doc_tower on all documents in the training set
def embed_passages(doc_tower, all_passages):
    # Get embeddings for all passages
    all_passages_tokens = []
    for passage in all_passages:
        passage_tokens = tokenize_text(passage)
        all_passages_tokens.append(passage_tokens)
    all_passage_embeddings = doc_tower(all_passages_tokens)
    
    # Create pairs of passages and their embeddings
    all_passages_and_embeddings = list(zip(all_passages, all_passage_embeddings))
    
    # Return the list of pairs
    print(f"All passages embedded")
    return all_passages_and_embeddings

# Upload all passage embeddings to vector database (Weaviate)
def upload_passages_to_vectordb(all_passages_and_embeddings):
    # Best practice: store your credentials in environment variables
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )

    print(client.is_ready())  # Should print: `True`

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    vector_db_name = f"doc_embeddings_avgpool_{current_time}"

    client.collections.create(vector_db_name)

    collection = client.collections.get(vector_db_name)

    with collection.batch.dynamic() as batch:
        for passage, embedding in all_passages_and_embeddings:
            # Convert the embedding tensor to a list
            if hasattr(embedding, 'detach'):
                embedding = embedding.detach().cpu().numpy().tolist()
            elif hasattr(embedding, 'numpy'):
                embedding = embedding.numpy().tolist()
            
            batch.add_object(
                properties={"text": passage},
                vector=embedding
            )
            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
                break

    failed_objects = collection.batch.failed_objects
    if failed_objects:
        print(f"Number of failed imports: {len(failed_objects)}")
        print(f"First failed object: {failed_objects[0]}")  
    print("Successfully added embeddings to vector db")
    client.close()

if __name__ == "__main__":
    doc_tower = load_doctower()
    all_passages = load_all_passages()
    all_passages_and_embeddings = embed_passages(doc_tower, all_passages)
    upload_passages_to_vectordb(all_passages_and_embeddings)