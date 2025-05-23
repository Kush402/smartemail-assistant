# fine-tune/dataset_loader.py

import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

def load_dataset_for_training(csv_path, tokenizer_name, test_size=0.1, max_length=512):
    """
    Load and prepare dataset for training.
    
    Args:
        csv_path (str): Path to the processed CSV file
        tokenizer_name (str): Name of the tokenizer to use
        test_size (float): Proportion of data to use for testing
        max_length (int): Maximum sequence length
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at {csv_path}")
    
    # Load data
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'processed_text' not in df.columns:
        raise ValueError("Dataset must contain 'processed_text' column")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples['processed_text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    print(f"✅ Dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    return train_dataset, test_dataset
