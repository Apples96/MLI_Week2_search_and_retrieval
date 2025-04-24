from B_dual_encoder_train import load_query_tower
import weaviate
from weaviate.classes.init import Auth
import torch
import os
from dotenv import load_dotenv
from A_dataset_fastbuild import tokenize_text

# Load environment variables
load_dotenv()



def get_query_embedding(query_text, query_tower):
    """Convert a query text into an embedding vector using the query tower."""
    # Tokenize the query
    query_tokens = tokenize_text(query_text)
    
    # Convert to batch format (list of token lists) as expected by the model
    batch_query_tokens = [query_tokens]
    
    # Generate embedding
    with torch.no_grad():
        query_embedding = query_tower(batch_query_tokens)
    
    # The result is a tensor with shape [1, embedding_dim]
    # Convert to a list for Weaviate
    query_embedding_list = query_embedding[0].detach().cpu().numpy().tolist()
    
    return query_embedding_list

def search_weaviate_cloud(query_text, collection_name, QUERY_TOWER_FILE_PATH, top_k=5):
    """Search for similar documents in Weaviate."""
    # Load environment variables
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    # Load query tower
    print("Loading query tower model...")
    query_tower, word2idx, idx2word = load_query_tower(filepath=QUERY_TOWER_FILE_PATH)
    
    # Get query embedding
    print(f"Generating embedding for query: '{query_text}'")
    query_embedding = get_query_embedding(query_text, query_tower)
    
    # Connect to Weaviate
    print("Connecting to Weaviate...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key)
    )
    
    print(f"Connected to Weaviate: {client.is_ready()}")
    
    # Get collection
    collection = client.collections.get(collection_name)
    
    # Search for similar vectors
    print(f"Searching for top {top_k} similar documents...")
    result = collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k,
    )
    
    # Process and return results
    search_results = []
    for i, obj in enumerate(result.objects):
        search_results.append({
            "rank": i + 1,
            "text": obj.properties.get("text", "No text available"),

            "score": obj.metadata.certainty if hasattr(obj.metadata, "certainty") else None,
            "id": obj.uuid
        })
    
    client.close()
    return search_results



def search_weaviate_local(query_text, collection_name, QUERY_TOWER_FILE_PATH, top_k=5):    
    # Load query tower
    print("Loading query tower model...")
    query_tower, word2idx, idx2word = load_query_tower(filepath=QUERY_TOWER_FILE_PATH)
    
    # Get query embedding
    print(f"Generating embedding for query: '{query_text}'")
    query_embedding = get_query_embedding(query_text, query_tower)
    
    # Connect to Weaviate
    print("Connecting to Weaviate...")
    client = weaviate.connect_to_local()
    
    print(f"Connected to Weaviate: {client.is_ready()}")
    
    # Get collection
    collection = client.collections.get(collection_name)
    
    # Search for similar vectors
    print(f"Searching for top {top_k} similar documents...")
    result = collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k,
    )
    
    # Process and return results
    search_results = []
    for i, obj in enumerate(result.objects):
        text = obj.properties.get("text", "No text available")
        if isinstance(text, list):
            text = " ".join(text)

        search_results.append({
            "rank": i + 1,
            "text": text,

            "score": obj.metadata.certainty if hasattr(obj.metadata, "certainty") else None,
            "id": obj.uuid
        })
    
    client.close()
    return search_results


def main():
    # Configuration
    #COLLECTION_NAME = "Doc_embeddings_avgpool_20250424_185546"  >> locally hosted, finetuned on hard negatives from model below. Model is : doctoweravg_hard.pt and qrytoweravg_hard.pt
    # QUERY_TOWER_FILE_PATH = "qrytoweravg_hard.pt"
    COLLECTION_NAME = "Doc_embeddings_avgpool_20250422_191349" # >> cloud hosted, trained on 2 epochs with random negatives. Model is : models/document_tower_avgpool.pt and models/query_tower_avgpool.pt
    QUERY_TOWER_FILE_PATH = "models/query_tower_avgpool.pt"
    TOP_K = 5  # Number of results to return
    
    # Example queries to test
    example_queries = [
        "how to not lose weight",
        # "what is the capital of France",
        # "worst recipes for chocolate cake",
        # "how does quantum computing work",
        # "symptoms of the common cold"
    ]
    
    # Run search for each query
    for query in example_queries:
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("="*80)
        
        results = search_weaviate_cloud(query, COLLECTION_NAME, QUERY_TOWER_FILE_PATH, TOP_K)
        #results = search_weaviate_local(query, COLLECTION_NAME, QUERY_TOWER_FILE_PATH, TOP_K)
        
        print(f"\nTop {TOP_K} results:")
        for result in results:
            print(f"\nRank {result['rank']}")
            # Print a snippet of the text (first 150 chars)
            text_snippet = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
            print(f"Text: {text_snippet}")
        
        print("\n" + "-"*80)

if __name__ == "__main__":
    main()