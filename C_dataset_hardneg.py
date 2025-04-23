import re
from tqdm import tqdm
import random
import torch
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
from torch.utils.data import Dataset
from A_dataset_fastbuild import tokenize_text, MSMarcoTripletDataset
import pickle

def process_hard_example(example):
    from evaluate_retrieval_model import search_weaviate
    """Process a single example for multiprocessing"""
    query = example["query"]
    query_id = example["query_id"]
    query_tokens = tokenize_text(query)
    
    processed_examples = []

    num_passages = len(example["passages"]["passage_text"])
    
    search_results = search_weaviate(query, collection_name = "Doc_embeddings_avgpool_20250422_191349", top_k=num_passages)
    neg_passages = [result["text"] for result in search_results]

    # Process each passage
    passages = example["passages"]
    for i, passage in enumerate(passages["passage_text"]):
        # Check if passage is selected by user, and tokenize
        is_selected = passages["is_selected"][i]
        pos_passage_tokens = tokenize_text(passage)
        neg_passage_tokens = tokenize_text(neg_passages[i])
        
        # Add to results
        processed_examples.append({
            "query_id": query_id,
            "query_tokens": query_tokens,
            "pos_passage_tokens": pos_passage_tokens,
            "is_selected": is_selected,
            "neg_passage_tokens": neg_passage_tokens,
        })
    
    return processed_examples

def load_and_preprocess_data_hard(filename = "hard_data", version="v1.1", split="validation", dataset_sample_size=100, n_workers=None):
    """
    Load MS MARCO dataset and preprocess it for the search engine task.
    
    Args:
        version: Dataset version ("v1.1" or "v2.1")
        split: Data split ("train", "validation", or "test")
        dataset_sample_size: Optional limit on number of examples to process
        n_workers: Number of workers for parallel processing (default: CPU count)
    
    Returns:
        Preprocessed data with tokenized queries and documents
    """
    print(f"Loading MS MARCO {version} {split} split...")
    dataset = load_dataset("microsoft/ms_marco", version)[split]

    
    if dataset_sample_size:
        # Only process a subset of data for testing
        dataset = dataset.select(range(min(dataset_sample_size, len(dataset))))
    
    # Process examples in parallel
    print(f"Processing examples using {n_workers} workers...")
    processed_data = []
    
    for example in tqdm(dataset, total = len(dataset), desc = "Preprocessing"):
        processed_examples = process_hard_example(example)
        processed_data.extend(processed_examples)
    
    # Print sample of the dataset
    random_indices = random.sample(range(len(processed_data)), min(10, len(processed_data)))
    print("\n10 random examples from the dataset:")
    for idx, rand_idx in enumerate(random_indices):
        example = processed_data[rand_idx]
        print(f"\nRandom Example {idx+1}:")
        print(f"Query ID: {example['query_id']}")
        print(f"Query tokens: {example['query_tokens'][:10]}...")
        print(f"Positive passage tokens: {example['pos_passage_tokens'][:10]}...")
        print(f"Is selected: {example['is_selected']}")
        print(f"Negative passage tokens: {example['neg_passage_tokens'][:10]}...")

    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    return MSMarcoTripletDataset(processed_data)

# Example usage:
# processed_data = load_and_preprocess_data(sample_size=1000, n_workers=8)

if __name__ == "__main__":
    processed_data = load_and_preprocess_data_hard(dataset_sample_size=100)
    print(f"Dataset size: {len(processed_data)}")