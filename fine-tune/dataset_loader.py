import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

def load_dataset_for_training(csv_path, tokenizer_name="gpt2", max_length=512):
    print(f"📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Create prompt and completion
    df["prompt"] = "Write a professional email response to:\nSubject: " + df["subject"] + "\n\nOriginal Email:\n" + df["body"] + "\n\nResponse:"
    
    # For now, we'll use a template response since we don't have actual responses
    df["completion"] = "Dear [Name],\n\nThank you for your email. I have received your request and will process it accordingly.\n\nBest regards,\n[Your name]"

    # Combine prompt and completion
    df["text"] = df["prompt"] + df["completion"]

    dataset = Dataset.from_pandas(df[["text"]])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Make sure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized_dataset, tokenizer
