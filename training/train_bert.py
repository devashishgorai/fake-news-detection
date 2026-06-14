from transformers import BertForSequenceClassification

print("Loading model...")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
)

print("SUCCESS!")