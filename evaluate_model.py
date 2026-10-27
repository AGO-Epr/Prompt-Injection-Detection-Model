
import re
import joblib
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score
)


model = joblib.load("models/prompt_injection_model.joblib")

wv = KeyedVectors.load("models/word2vec_model.kv")


def get_sentence_vector(text, wv):
    words = re.findall(r"[A-Za-z0-9_]+", str(text))
    vectors = []

    for word in words:
        candidates = [word, word.lower(), word.capitalize()]

        for candidate in candidates:
            if candidate in wv:
                vectors.append(wv[candidate])
                break

    if not vectors:
        return np.zeros(wv.vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)



df = pd.read_csv("dataset.csv")
df = df[["text", "label"]].dropna()

df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

df["label"] = pd.to_numeric(df["label"], errors="raise")

if not df["label"].isin([0, 1]).all():
    raise ValueError("Labels must be 0 or 1.")

df["label"] = df["label"].astype(int)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("Dataset shape:", df.shape)


X = np.vstack([
    get_sentence_vector(text, wv)
    for text in df["text"]
])

y = df["label"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

y_pred = model.predict(X_test)


train_pred = model.predict(X_train)

print(
    "Training Accuracy:",
    f"{accuracy_score(y_train, train_pred):.2%}"
)

print(
    "Testing Accuracy:",
    f"{accuracy_score(y_test, y_pred):.2%}"
)

accuracy = accuracy_score(y_test, y_pred)

injection_precision = precision_score(
    y_test, y_pred, pos_label=1, zero_division=0
)

injection_recall = recall_score(
    y_test, y_pred, pos_label=1, zero_division=0
)

injection_f1 = f1_score(
    y_test, y_pred, pos_label=1, zero_division=0
)


print("\n========== MODEL METRICS ==========")

print(f"Accuracy:            {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Injection Precision: {injection_precision:.4f} ({injection_precision * 100:.2f}%)")
print(f"Injection Recall:    {injection_recall:.4f} ({injection_recall * 100:.2f}%)")
print(f"Injection F1-score:  {injection_f1:.4f} ({injection_f1 * 100:.2f}%)")


print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Benign", "Injection"],
    zero_division=0,
    digits=4
))


cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\n========== CONFUSION MATRIX ==========")
print("                 Predicted")
print("                 Benign  Injection")
print(f"Actual Benign    {cm[0, 0]:6d}  {cm[0, 1]:9d}")
print(f"Actual Injection {cm[1, 0]:6d}  {cm[1, 1]:9d}")

print("\nFalse positives:", cm[0, 1])
print("False negatives:", cm[1, 0])

print("\nEvaluation completed!")
