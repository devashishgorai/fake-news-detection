import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

fake = pd.read_csv(DATASET_DIR / "Fake.csv")
true = pd.read_csv(DATASET_DIR / "True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Merge datasets
df = pd.concat([fake, true], ignore_index=True)

print("Total Records:", len(df))

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Preview:")
print(df.head())
