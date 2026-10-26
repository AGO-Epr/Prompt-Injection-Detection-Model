
import re
import numpy as np
import pandas as pd
import gensim.downloader as api
import joblib

import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ==========================================
# 1. Load the dataset
# ==========================================

print("Loading dataset.csv...")

df = pd.read_csv("dataset.csv")

# Keep only the required columns
df = df[["text", "label"]]

# Clean missing and empty values
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

# Ensure labels are numeric
df["label"] = pd.to_numeric(df["label"], errors="raise")

if not df["label"].isin([0, 1]).all():
    raise ValueError("Labels must be 0 or 1.")

df["label"] = df["label"].astype(int)

# Remove duplicate prompts
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("Dataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())

# Confirm label meanings before training:
# 0 = benign
# 1 = prompt injection

# ==========================================
# 2. Load pretrained Word2Vec
# ==========================================

print("\nLoading Word2Vec model...")
print("The first download may take some time.")

wv = api.load("word2vec-google-news-300")

print("Word2Vec loaded!")
print("Vector dimensions:", wv.vector_size)

# ==========================================
# 3. Convert each prompt to a 300-D vector
# ==========================================

def get_sentence_vector(text, wv):
    # Tokenize while retaining original capitalization
    words = re.findall(r"[A-Za-z0-9_]+", text)

    vectors = []

    for word in words:
        # Google News Word2Vec is case-sensitive.
        # Try original, lowercase and capitalized forms.
        candidates = [
            word,
            word.lower(),
            word.capitalize()
        ]

        for candidate in candidates:
            if candidate in wv:
                vectors.append(wv[candidate])
                break

    # Return a zero vector if no words are in vocabulary
    if not vectors:
        return np.zeros(wv.vector_size, dtype=np.float32)

    # Average word embeddings into one prompt vector
    return np.mean(vectors, axis=0).astype(np.float32)


print("\nConverting prompts into vectors...")

X = np.vstack([
    get_sentence_vector(text, wv)
    for text in df["text"]
])

y = df["label"].to_numpy()

print("Feature matrix shape:", X.shape)
print("Label array shape:", y.shape)

# Expected feature shape: (number_of_prompts, 300)

# ==========================================
# 4. Split into training and testing data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 5. Train the classifier
# ==========================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Training complete!")

# ==========================================
# 6. Evaluate the model
# ==========================================

y_pred = model.predict(X_test)



print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Injection"],
        zero_division=0
    )
)


# Create the models folder
os.makedirs("models", exist_ok=True)

# Save the Logistic Regression classifier
joblib.dump(model, "models/prompt_injection_model.joblib")

# Save the Word2Vec embeddings
wv.save("models/word2vec_model.kv")

print("Models saved successfully!")