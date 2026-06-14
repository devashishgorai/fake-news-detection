import os
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR)

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models" / "fake_news_bert"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets():
    print("Loading dataset...")

    df = pd.read_csv(DATASET_DIR / "cleaned_news.csv")
    df["content"] = df["content"].fillna("")

    print("Splitting data...")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["content"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    return train_texts, test_texts, train_labels, test_labels


def load_tokenizer():
    print("Loading tokenizer...")

    return BertTokenizer.from_pretrained(
        "bert-base-uncased",
        cache_dir=str(CACHE_DIR),
    )


def build_model():
    print("Loading model...")

    print("Transformers import works")
    print("About to load bert-base-uncased")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    print("Model loaded successfully!")

    return model


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def tokenize_dataset(
    tokenizer,
    train_texts,
    test_texts,
    train_labels,
    test_labels,
):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    train_dataset = Dataset.from_dict(
        {
            "text": train_texts.tolist(),
            "label": train_labels.tolist(),
        }
    )

    test_dataset = Dataset.from_dict(
        {
            "text": test_texts.tolist(),
            "label": test_labels.tolist(),
        }
    )

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    return train_dataset, test_dataset


def run_training():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print("Using Device:", device)

    train_texts, test_texts, train_labels, test_labels = load_datasets()

    tokenizer = load_tokenizer()

    train_dataset, test_dataset = tokenize_dataset(
        tokenizer,
        train_texts,
        test_texts,
        train_labels,
        test_labels,
    )

    model = build_model()

    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        logging_steps=100,
        logging_strategy="steps",
        load_best_model_at_end=True,
        dataloader_pin_memory=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting BERT Training...\n")

    trainer.train()

    print("\nSaving Model...\n")

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    print("\nModel saved successfully!")
    print("Location:", MODEL_DIR)

    results = trainer.evaluate()

    print("\nFinal Results:")
    print(results)


def main():
    run_training()


if __name__ == "__main__":
    main()
