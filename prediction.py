
import re
import joblib
import numpy as np

from gensim.models import KeyedVectors


# ==========================================
# 1. Load saved models
# ==========================================

print("Loading saved models...")

model = joblib.load("models/prompt_injection_model.joblib")

wv = KeyedVectors.load("models/word2vec_model.kv")

print("Models loaded successfully!")


# ==========================================
# 2. Convert prompt into a 300-D vector
# ==========================================

def get_sentence_vector(text, wv):
    words = re.findall(r"[A-Za-z0-9_]+", text)
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


# ==========================================
# 3. Predict a prompt
# ==========================================

def predict_prompt(text):

    vector = get_sentence_vector(text, wv)
    vector = vector.reshape(1, -1)

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    class_index = list(model.classes_).index(prediction)
    confidence = probabilities[class_index]

    if prediction == 1:
        result = "POTENTIAL PROMPT INJECTION"
    else:
        result = "BENIGN"

    return result, confidence


# ==========================================
# 4. Take user input
# ==========================================

print("\n========== PROMPT INJECTION DETECTOR ==========")
print("Enter 'exit' to quit.")

while True:

    prompt = input("\nEnter a prompt: ").strip()

    if prompt.lower() == "exit":
        print("Exiting detector.")
        break

    if not prompt:
        print("Please enter a non-empty prompt.")
        continue

    result, confidence = predict_prompt(prompt)

    print("\nPrediction:", result)
    print(f"Model confidence: {confidence:.2%}")
