# data/prepare_data.py
import pandas as pd
import os
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).parent
RAW_PATH = BASE_DIR / "raw" / "raw_emails.csv"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "processed_emails.csv"

def preprocess_emails(df):
    """Preprocess the email data."""
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Add email length
    df['body_length'] = df['body'].str.len()
    
    # Add word count
    df['word_count'] = df['body'].str.split().str.len()
    
    # Clean up whitespace
    df['body'] = df['body'].str.strip()
    df['subject'] = df['subject'].str.strip()
    
    return df

def main():
    # Create processed directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Read raw data
    print(f"Reading data from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    
    # Preprocess data
    print("Preprocessing data...")
    processed_df = preprocess_emails(df)
    
    # Save processed data
    print(f"Saving processed data to {PROCESSED_PATH}")
    processed_df.to_csv(PROCESSED_PATH, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
