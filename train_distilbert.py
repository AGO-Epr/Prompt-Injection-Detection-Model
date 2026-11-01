
import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_NAME = "distilbert/distilbert-base-uncased"

DATASET_PATH = "dataset.csv"
OUTPUT_DIR = "models/distilbert_final"

MAX_LENGTH = 256
BATCH_SIZE = 4
MAX_EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

RANDOM_STATE = 42

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("FINAL DISTILBERT TRAINING")
print("=" * 60)

print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==================================================
# 2. LOAD AND CLEAN DATASET
# ==================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

if not {"text", "label"}.issubset(df.columns):
    raise ValueError("Dataset must contain text and label columns.")

df = df[["text", "label"]].copy()

df["text"] = df["text"].fillna("").astype(str).str.strip()

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df.dropna(subset=["label"])
df = df[df["text"] != ""]

if not df["label"].isin([0, 1]).all():
    raise ValueError("Labels must only contain 0 and 1.")

df["label"] = df["label"].astype(int)

# Remove exact duplicate texts
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

if df["label"].nunique() != 2:
    raise ValueError("Both labels 0 and 1 must be present.")

print("Dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())


# ==================================================
# 3. SPLIT INTO DEVELOPMENT AND TEST DATA
# ==================================================

print("\nSplitting dataset...")

development_df, test_df = train_test_split(
    df,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

print("Development samples:", len(development_df))
print("Test samples:", len(test_df))

# Save the untouched test set
os.makedirs("data", exist_ok=True)

test_df.to_csv(
    "data/distilbert_final_test.csv",
    index=False
)


# ==================================================
# 4. CREATE TRAINING AND VALIDATION DATA
# ==================================================

# Validation is used only to select the best epoch.

train_df, val_df = train_test_split(
    development_df,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=development_df["label"]
)

print("\nDevelopment stage split:")
print("Training:", len(train_df))
print("Validation:", len(val_df))


# ==================================================
# 5. LOAD TOKENIZER
# ==================================================

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ==================================================
# 6. TOKENIZATION FUNCTION
# ==================================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


def prepare_dataset(dataframe):

    dataset = Dataset.from_pandas(
        dataframe[["text", "label"]].reset_index(drop=True)
    )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    dataset = dataset.rename_column(
        "label",
        "labels"
    )

    return dataset


train_dataset = prepare_dataset(train_df)
val_dataset = prepare_dataset(val_df)
test_dataset = prepare_dataset(test_df)

# Final training dataset includes training + validation data.
final_train_dataset = prepare_dataset(development_df)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ==================================================
# 7. MODEL CREATION FUNCTION
# ==================================================

def create_model():

    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={
            0: "BENIGN",
            1: "PROMPT_INJECTION"
        },
        label2id={
            "BENIGN": 0,
            "PROMPT_INJECTION": 1
        }
    )


# ==================================================
# 8. EVALUATION METRICS
# ==================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ==================================================
# 9. DEVELOPMENT TRAINING
# ==================================================

print("\n" + "=" * 60)
print("STAGE 1: FINDING THE BEST TRAINING EPOCH")
print("=" * 60)

development_model = create_model()

development_args = TrainingArguments(
    output_dir="training_output/development",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=4,

    num_train_epochs=MAX_EPOCHS,

    weight_decay=WEIGHT_DECAY,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    fp16=(device == "cuda"),

    logging_steps=100,

    save_total_limit=2,

    report_to="none",

    seed=RANDOM_STATE,

    dataloader_pin_memory=(device == "cuda")
)

development_trainer = Trainer(
    model=development_model,
    args=development_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=1
        )
    ]
)

print("\nStarting development training...")

development_trainer.train()


# ==================================================
# 10. SELECT BEST EPOCH
# ==================================================

eval_history = [
    entry
    for entry in development_trainer.state.log_history
    if "eval_f1" in entry and "epoch" in entry
]

if not eval_history:
    raise RuntimeError("Validation metrics were not recorded.")

best_result = max(
    eval_history,
    key=lambda entry: entry["eval_f1"]
)

best_epoch = max(
    1,
    int(np.ceil(best_result["epoch"]))
)

print("\nBest validation F1:", best_result["eval_f1"])
print("Selected training epochs:", best_epoch)


# ==================================================
# 11. FINAL TRAINING ON 90% DATA
# ==================================================

print("\n" + "=" * 60)
print("STAGE 2: FINAL MODEL TRAINING")
print("=" * 60)

print("Training samples:", len(final_train_dataset))
print("Epochs:", best_epoch)

# Start with fresh pretrained DistilBERT weights.
# Do not reuse the development model's weights.

final_model = create_model()

final_args = TrainingArguments(
    output_dir="training_output/final",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=4,

    num_train_epochs=best_epoch,

    weight_decay=WEIGHT_DECAY,

    fp16=(device == "cuda"),

    logging_steps=100,

    save_strategy="no",

    report_to="none",

    seed=RANDOM_STATE,

    dataloader_pin_memory=(device == "cuda")
)

final_trainer = Trainer(
    model=final_model,
    args=final_args,

    train_dataset=final_train_dataset,

    processing_class=tokenizer,

    data_collator=data_collator
)

print("\nTraining final model...")

final_trainer.train()


# ==================================================
# 12. SAVE FINAL MODEL
# ==================================================

print("\nSaving final DistilBERT model...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

final_trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("Model saved at:", OUTPUT_DIR)


# ==================================================
# 13. EVALUATE ON UNSEEN TEST DATA
# ==================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_results = final_trainer.predict(test_dataset)

predictions = np.argmax(
    test_results.predictions,
    axis=-1
)

true_labels = test_results.label_ids


# ==================================================
# 14. PRINT CLASSIFICATION REPORT
# ==================================================

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predictions,
        labels=[0, 1],
        target_names=[
            "Benign",
            "Prompt Injection"
        ],
        zero_division=0
    )
)


# ==================================================
# 15. CONFUSION MATRIX
# ==================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=[0, 1]
)

print(cm)


# ==================================================
# 16. FINAL METRICS
# ==================================================

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predictions,
    average="binary",
    pos_label=1,
    zero_division=0
)

accuracy = accuracy_score(
    true_labels,
    predictions
)

print("\nFINAL MODEL PERFORMANCE")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


# ==================================================
# 17. SAVE TEST PREDICTIONS
# ==================================================

test_results_df = test_df.copy()

test_results_df["predicted_label"] = predictions

test_results_df.to_csv(
    "data/distilbert_final_predictions.csv",
    index=False
)

print("\nTest predictions saved.")

print("\n" + "=" * 60)
print("FINAL DISTILBERT TRAINING COMPLETED")
print("=" * 60)
