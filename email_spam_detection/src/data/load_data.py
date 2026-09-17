import pandas as pd


def load_data(path):
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        print("UTF-8 failed. Loading with Latin-1 encoding...")
        df = pd.read_csv(path, encoding="latin1")

    # Rename original dataset columns
    df = df.rename(columns={
        "v1": "label",
        "v2": "message"
    })

    # Keep only required columns
    df = df[["label", "message"]]

    return df