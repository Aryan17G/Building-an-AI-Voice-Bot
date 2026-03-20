import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Define 10 Customer Support Intents & Sample Data
# In a real scenario, load this from a CSV file.
data = {
    "text": [
        "Where is my package?", "Track my order please.", "When will my item arrive?",
        "I want to cancel my order", "Please cancel my purchase", "Stop my shipment",
        "I need a refund", "Give me my money back", "How do I get a refund?",
        "My payment failed", "Credit card declined", "Issue with my payment",
        "I want to return this item", "How do I send this back?", "Return policy?",
        "Update my shipping address", "Wrong delivery address", "Change my shipping details",
        "Cancel my subscription", "End my membership", "Stop charging my card monthly",
        "I forgot my password", "Reset my account password", "Can't log in",
        "Is this product in stock?", "Tell me more about this item", "Product specifications",
        "Let me talk to a human", "Speak to a real agent", "Transfer me to customer service"
    ],
    "label_name": [
        "order_status", "order_status", "order_status",
        "cancel_order", "cancel_order", "cancel_order",
        "refund_request", "refund_request", "refund_request",
        "payment_issue", "payment_issue", "payment_issue",
        "return_item", "return_item", "return_item",
        "change_address", "change_address", "change_address",
        "cancel_subscription", "cancel_subscription", "cancel_subscription",
        "reset_password", "reset_password", "reset_password",
        "product_info", "product_info", "product_info",
        "speak_to_agent", "speak_to_agent", "speak_to_agent"
    ]
}

df = pd.DataFrame(data)

# Map labels to integers
labels = df['label_name'].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
df['label'] = df['label_name'].map(label2id)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# 2. Tokenization
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create PyTorch Dataset
class SupportDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SupportDataset(train_encodings, train_labels)
val_dataset = SupportDataset(val_encodings, val_labels)

# 3. Model Setup
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
)

# 4. Define Metrics Calculation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,              
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",       # <--- NEW PARAMETER
    logging_dir='./logs',
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 6. Train and Save
print("Starting training...")
trainer.train()

# Save the fine-tuned model and tokenizer for your FastAPI app
model.save_pretrained("./support-intent-model")
tokenizer.save_pretrained("./support-intent-model")
print("Model saved to ./support-intent-model")

# 7. Generate Evaluation Metrics and Confusion Matrix
print("Evaluating...")
predictions = trainer.predict(val_dataset)
preds = predictions.predictions.argmax(-1)

# Print Metrics
print("\n--- Final Metrics ---")
for key, val in predictions.metrics.items():
    print(f"{key}: {val:.4f}")

# Plot Confusion Matrix
cm = confusion_matrix(val_labels, preds, labels=list(range(len(labels))))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Intent')
plt.ylabel('Actual Intent')
plt.title('Intent Classification Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to confusion_matrix.png")