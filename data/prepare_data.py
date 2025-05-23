# data/prepare_data.py

import os
import pandas as pd

RAW_DATA_PATH = "data/raw/raw_emails.csv"
PROCESSED_DATA_PATH = "data/processed/processed_emails.csv"

def preprocess_emails(df):
    # Drop rows with missing fields
    df = df.dropna(subset=["subject", "body"])

    # Remove leading/trailing whitespace
    df["subject"] = df["subject"].str.strip()
    df["body"] = df["body"].str.strip()

    # Calculate body length and word count
    df["body_length"] = df["body"].str.len()
    df["word_count"] = df["body"].str.split().str.len()

    # Keep only required columns
    df = df[["subject", "body", "body_length", "word_count"]]

    return df

def main():
    print(f"Reading data from {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Preprocessing data...")
    processed_df = preprocess_emails(df)

    print(f"Saving processed data to {PROCESSED_DATA_PATH}")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("✅ Done!")

if __name__ == "__main__":
    main()
