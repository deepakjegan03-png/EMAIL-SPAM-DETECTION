import pandas as pd
from src.data.preprocess import clean_batch


def process_dataset():

    print("Script Started")

    # Load dataset with compatible encoding
    df = pd.read_csv(
        "data/raw/email_spam.csv",
        encoding="latin1"
    )

    print("Dataset Loaded")

    # Rename dataset columns to match project requirements
    df = df.rename(columns={
        "v1": "label",
        "v2": "email_text"
    })

    # Lowercasing + regex are fast as vectorized pandas ops,
    # so we do them here and leave only tokenizing/lemmatizing to spaCy.

    texts = df["email_text"].astype(str).str.lower()
    texts = texts.str.replace(r"http\S+", "", regex=True)
    texts = texts.str.replace(r"[^a-zA-Z ]", "", regex=True)

    df["clean_email"] = clean_batch(texts)

    print("Cleaning Completed")

    df.to_csv("data/processed/cleaned_emails.csv", index=False)

    print("File Saved")
    

