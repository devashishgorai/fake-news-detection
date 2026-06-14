import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

# Load cleaned dataset
df = pd.read_csv(DATASET_DIR / "cleaned_news.csv")
df["content"] = df["content"].fillna("")

print("Dataset Shape:", df.shape)

# Features and Labels
X = df["content"]
y = df["label"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

# Predictions
predictions = model.predict(X_test_tfidf)

# Metrics
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(report)
