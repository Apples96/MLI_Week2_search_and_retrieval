import torch
from datasets import load_dataset
import weaviate
from weaviate.classes.init import Auth
import os
from tqdm import tqdm
from A_dataset_fastbuild import tokenize_text
from B_dual_encoder_train import DocTower, CBOW, load_cbow_embedding
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load document_tower model
def load_doctower():
    print("Loading DocTower model...")
    checkpoint = torch.load("doctoweravg_hard.pt", map_location=torch.device('cpu'))
    embedding, word2idx, idx2word, embedding_dim = load_cbow_embedding()
    doc_tower = DocTower(embedding, embedding_dim, word2idx)
    doc_tower.load_state_dict(checkpoint['model_state_dict'])
    doc_tower.eval()
    print("DocTower model loaded successfully")
    return doc_tower

# Load a subset of passages with progress tracking
def load_passages(max_passages=10000, dataset_splits=['validation']):
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    all_passages = []
    count = 0

    # Use tqdm for progress tracking
    for split_name in tqdm(dataset_splits, desc="Processing splits"):
        split_dataset = dataset[split_name]
        
        # Create progress bar for each split
        for item in tqdm(split_dataset, desc=f"Processing {split_name}", leave=False):
            passage_texts = item['passages']['passage_text']
                
            # Add passages to our collection
            for passage in passage_texts:
                passage_tokens = tokenize_text(passage)
                all_passages.append(passage_tokens)
                count += 1
                if max_passages is not None and count >= max_passages:
                    print(f"Loaded {len(all_passages)} passages (limited to {max_passages})")
                    return all_passages

    print(f"Total passages collected: {len(all_passages)}")
    return all_passages

def embed_passages(doc_tower, all_passages):
    with torch.no_grad():
        passages_embeddings = doc_tower(all_passages)
    passages_and_embeddings = list(zip(passages_embeddings, all_passages))
    return passages_and_embeddings


# Embed passages in batches with progress tracking
def upload_cloud(passages_and_embeddings, batch_size=64, collection_name=None):
    # Get environment variables
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not weaviate_url or not weaviate_api_key:
        print("Error: WEAVIATE_URL or WEAVIATE_API_KEY environment variables are not set")
        return
    
    # Create connection
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )
    
    print(f"Connected to Weaviate: {client.is_ready()}")
    
    # Generate a collection name if not provided
    if collection_name is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        collection_name = f"doc_embeddings_avgpool_{current_time}"
    
    # Create collection
    # Try to create the collection directly, and handle the case if it already exists
    try:
        print(f"Creating collection: {collection_name}")
        client.collections.create(collection_name)
        print(f"Created new collection: {collection_name}")
    except Exception as e:
        # If the collection already exists, this will cause an exception
        if "already exists" in str(e).lower():
            print(f"Collection {collection_name} already exists, using existing collection")
        else:
            print(f"Error creating collection: {str(e)}")
            client.close()
            return
        
    collection = client.collections.get(collection_name)
    
    # Calculate number of batches
    total_batches = (len(passages_and_embeddings) - 1) // batch_size + 1
    print(f"Processing {len(passages_and_embeddings)} passages in {total_batches} batches")
    
    # Process in batches
    processed_count = 0
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(passages_and_embeddings))
        
        # Progress information
        print(f"Processing batch {batch_idx+1}/{total_batches} ({start_idx}-{end_idx-1} of {len(passages_and_embeddings)} passages)")
        
        # Upload this batch immediately
        print("Uploading to Weaviate...")
        with collection.batch.dynamic() as batch:
            batch_items = passages_and_embeddings[start_idx:end_idx]
            for idx, (embedding, passage) in enumerate(batch_items):
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
        
        processed_count += batch_size
        print(f"Progress: {processed_count}/{len(passages_and_embeddings)} passages processed ({processed_count/len(passages_and_embeddings)*100:.1f}%)")
        
        # Check for failures
        failed_objects = collection.batch.failed_objects
        if failed_objects:
            print(f"Number of failed imports in this batch: {len(failed_objects)}")
    
    print(f"Successfully processed and uploaded {processed_count} passages into {collection_name}")
    client.close()
    print("Weaviate connection closed")




def upload_local(passages_and_embeddings, batch_size=64, collection_name=None):
    # Get environment variables
    client = weaviate.connect_to_local()
    
    print(f"Connected to Weaviate: {client.is_ready()}")
    
    # Generate a collection name if not provided
    if collection_name is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        collection_name = f"doc_embeddings_avgpool_{current_time}"
    
    # Create collection
    # Try to create the collection directly, and handle the case if it already exists
    try:
        print(f"Creating collection: {collection_name}")
        client.collections.create(collection_name)
        print(f"Created new collection: {collection_name}")
    except Exception as e:
        # If the collection already exists, this will cause an exception
        if "already exists" in str(e).lower():
            print(f"Collection {collection_name} already exists, using existing collection")
        else:
            print(f"Error creating collection: {str(e)}")
            client.close()
            return
        
    collection = client.collections.get(collection_name)
    
    # Calculate number of batches
    total_batches = (len(passages_and_embeddings) - 1) // batch_size + 1
    print(f"Processing {len(passages_and_embeddings)} passages in {total_batches} batches")
    
    # Process in batches
    processed_count = 0
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(passages_and_embeddings))
        
        # Progress information
        print(f"Processing batch {batch_idx+1}/{total_batches} ({start_idx}-{end_idx-1} of {len(passages_and_embeddings)} passages)")
        
        # Upload this batch immediately
        print("Uploading to Weaviate...")
        with collection.batch.dynamic() as batch:
            batch_items = passages_and_embeddings[start_idx:end_idx]
            for idx, (embedding, passage) in enumerate(batch_items):
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
        
        processed_count += batch_size
        print(f"Progress: {processed_count}/{len(passages_and_embeddings)} passages processed ({processed_count/len(passages_and_embeddings)*100:.1f}%)")
        
        # Check for failures
        failed_objects = collection.batch.failed_objects
        if failed_objects:
            print(f"Number of failed imports in this batch: {len(failed_objects)}")
    
    print(f"Successfully processed and uploaded {processed_count} passages into {collection_name}")
    client.close()
    print("Weaviate connection closed")




def main():
    try:
        # Parameters
        MAX_PASSAGES = None  # Set to None for all passages, or a number for a subset
        BATCH_SIZE = 1000       # Batch size for processing
        
        # Only process validation set for testing (much smaller than train)
        DATASET_SPLITS = ['validation']  # Use ['train', 'validation', 'test'] for all data
        
        # Collection name
        collection_name = None  # Change as needed
        
        # Step 1: Load model
        doc_tower = load_doctower()
        
        # Step 2: Load passages (limited set for testing)
        passages = load_passages(max_passages=MAX_PASSAGES, dataset_splits=DATASET_SPLITS)

        passages_and_embeddings = embed_passages(doc_tower, passages)

        # Step 3: Process and upload in a streaming fashion (no pickle files)
        # upload_cloud(passages_and_embeddings, batch_size=BATCH_SIZE, collection_name=collection_name)
        upload_local(passages_and_embeddings, batch_size=BATCH_SIZE, collection_name=collection_name)
        
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

