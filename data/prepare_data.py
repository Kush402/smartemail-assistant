# data/prepare_data.py

import os
import pandas as pd

RAW_DATA_PATH = "data/raw/raw_emails.csv"
PROCESSED_DATA_PATH = "data/processed/processed_emails.csv"

def preprocess_emails(df):
    # Drop rows with missing fields
    df = df.dropna(subset=["subject", "body", "response"])

    # Remove leading/trailing whitespace
    df["subject"] = df["subject"].str.strip()
    df["body"] = df["body"].str.strip()
    df["response"] = df["response"].str.strip()

    # Format prompt
    df["prompt"] = (
        "You are an HR manager. Write a professional response to the following email:\n\n"
        "Subject: " + df["subject"] + "\n\n"
        "Email:\n" + df["body"] + "\n\n"
        "Response:"
    )

    # Rename response to completion
    df = df[["prompt", "response"]]
    df.rename(columns={"response": "completion"}, inplace=True)

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
