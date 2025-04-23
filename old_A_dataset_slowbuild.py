import re
import numpy as np
from datasets import load_dataset
import torch
from tqdm import tqdm
import random



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

def load_and_preprocess_data(version="v1.1", split="train"):
    """
    Load MS MARCO dataset and preprocess it for the search engine task.
    
    Args:
        version: Dataset version ("v1.1" or "v2.1")
        split: Data split ("train", "validation", or "test")
    
    Returns:
        Preprocessed data with tokenized queries and documents
    """
    print(f"Loading MS MARCO {version} {split} split...")
    dataset = load_dataset("microsoft/ms_marco", version)[split]
    
    # Create lists to store processed data
    processed_data = {
        "query_id": [],
        "query_tokens": [],
        "pos_passage_tokens": [],
        "is_selected": [],
        "neg_passage_tokens": [],
    }
    
    # Process each example
    for example in tqdm(dataset, desc="Preprocessing"):
        query = example["query"]
        query_id = example["query_id"]
        query_tokens = tokenize_text(query)
        
        # Process each passage
        passages = example["passages"]
        for i, passage in enumerate(passages["passage_text"]):
            # Check if Bing passage is selected by user, and tokenise passage
            is_selected = passages["is_selected"][i]
            pos_passage_tokens = tokenize_text(passage)
                
            # Pick a random negative example
            random_example_idx = torch.randint(0, len(dataset["query_id"]), (1,)).item()
            neg_example = dataset[random_example_idx]
            random_passage_idx = torch.randint(0, len(neg_example["passages"]["passage_text"]), (1,)).item()
            neg_passage = neg_example["passages"]["passage_text"][random_passage_idx]
            neg_passage_tokens = tokenize_text(neg_passage)
            
            # Add new example to dataset
            processed_data["query_id"].append(query_id)
            processed_data["query_tokens"].append(query_tokens)
            processed_data["pos_passage_tokens"].append(pos_passage_tokens)
            processed_data["is_selected"].append(is_selected)
            processed_data["neg_passage_tokens"].append(neg_passage_tokens)
    
    # Get total number of examples
    total_examples = len(processed_data["query_id"])

    # Generate 10 random indices
    random_indices = random.sample(range(total_examples), min(10, total_examples))

    # Print 10 random examples
    print("\n10 random examples from the dataset:")
    for idx, rand_idx in enumerate(random_indices):
        print(f"\nRandom Example {idx+1}:")
        print(f"Query ID: {processed_data['query_id'][rand_idx]}")
        print(f"Query: {processed_data['query'][rand_idx]}")
        print(f"Passage text: {processed_data['passage_text'][rand_idx][:100]}..." if len(processed_data['passage_text'][rand_idx]) > 100 else processed_data['passage_text'][rand_idx])
        print(f"Is selected: {processed_data['is_selected'][rand_idx]}")
    return processed_data

if __name__ == "__main__":
    processed_data = load_and_preprocess_data()
