import re
from tqdm import tqdm
import random
import torch
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
from torch.utils.data import Dataset
from B_dual_encoder_training import DualEncoder

# Pre-compile regex pattern - major speedup!

def tokenize_text(text):
    """
    Tokenize text by:
    1. Converting to lowercase
    2. Replacing non-alphanumeric characters with spaces
    3. Splitting on whitespace
    
    Returns a list of tokens (words).
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace any character that is not a letter or number with a space
    text = re.sub(r'[^a-z0-9]', ' ', text)
    
    # Split on whitespace and filter empty tokens
    tokens = [token for token in text.split() if token]
    
    return tokens

def process_example(example):
    """Process a single example for multiprocessing"""
    
    query = example["query"]
    query_id = example["query_id"]
    query_tokens = tokenize_text(query)
    
    processed_examples = []
    
    # Process each passage
    passages = example["passages"]
    for i, passage in enumerate(passages["passage_text"]):
        # Check if passage is selected by user, and tokenize
        if passages["is_selected"][i] == 1:
            pos_passage_tokens = tokenize_text(passage)
            # Generate a hard negative  example from the current model
            neg_passage = random.choice(neg_passages_pool)
            neg_passage_tokens = tokenize_text(neg_passage)
        
        neg_passage_tokens = tokenize_text(neg_passage)
        
        # Add to results
        processed_examples.append({
            "query_id": query_id,
            "query_tokens": query_tokens,
            "pos_passage_tokens": pos_passage_tokens,
            "is_selected": is_selected,
            "neg_passage_tokens": neg_passage_tokens,
        })
    
    return processed_examples

class MSMarcoTripletDatasetHardNeg(Dataset):
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def load_and_preprocess_data(version="v1.1", split="train", dataset_sample_size=100000, n_workers=None):
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
    model = DualEncoder()
    model.load_state_dict(torch.load('DualEncoder.pt'))
    model.eval()
    
    if dataset_sample_size:
        # Only process a subset of data for testing
        dataset = dataset.select(range(min(dataset_sample_size, len(dataset))))
    
    # Set up multiprocessing
    if n_workers is None:
        n_workers = cpu_count()
    
    # Process examples in parallel
    print(f"Processing examples using {n_workers} workers...")
    processed_data = []
    with Pool(n_workers) as pool:
        # Process examples in chunks to show progress
        chunk_size = max(1, len(dataset) // (n_workers * 10))
        for chunk_results in tqdm(
            pool.imap(process_example, chunksize=chunk_size),
            total=len(dataset),
            desc="Preprocessing"
        ):
            processed_data.extend(chunk_results)
    
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
    
    return MSMarcoTripletDatasetHardNeg(processed_data)

# Example usage:
# processed_data = load_and_preprocess_data(sample_size=1000, n_workers=8)

if __name__ == "__main__":
    processed_data = load_and_preprocess_data(dataset_sample_size=100)
    print(f"Dataset size: {len(processed_data)}")