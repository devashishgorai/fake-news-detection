import pandas as pd
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

# Load datasets
fake = pd.read_csv(DATASET_DIR / "Fake.csv")
true = pd.read_csv(DATASET_DIR / "True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Merge
df = pd.concat([fake, true], ignore_index=True)

# Use title + text together
df["content"] = df["title"] + " " + df["text"]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove symbols
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text.strip()


df["content"] = df["content"].apply(clean_text)

# Keep only what we need
df = df[["content", "label"]]

print(df.head())
print("\nDataset Shape:", df.shape)

# Save cleaned dataset
df.to_csv(DATASET_DIR / "cleaned_news.csv", index=False)

print("\nSaved as cleaned_news.csv")
import pandas as pd
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

# Load datasets
fake = pd.read_csv(DATASET_DIR / "Fake.csv")
true = pd.read_csv(DATASET_DIR / "True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Merge
df = pd.concat([fake, true], ignore_index=True)

# Use title + text together
df["content"] = df["title"] + " " + df["text"]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove symbols
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text.strip()


df["content"] = df["content"].apply(clean_text)

# Keep only what we need
df = df[["content", "label"]]

print(df.head())
print("\nDataset Shape:", df.shape)

# Save cleaned dataset
df.to_csv(DATASET_DIR / "cleaned_news.csv", index=False)

print("\nSaved as cleaned_news.csv")
