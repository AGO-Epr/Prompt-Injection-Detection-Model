# Prompt Injection Detection Model

A machine learning project that classifies text as **Benign** or **Prompt Injection** using Word2Vec embeddings and Logistic Regression.

## About the Project

This project aims to detect prompts that attempt to manipulate or override an AI model's instructions.

- `0` — Benign text
- `1` — Prompt injection

## Dataset

- **File:** `dataset.csv`
- **Total samples:** 29,019
- **Columns:** `text` and `label`
- **Label 0:** Benign text
- **Label 1:** Prompt injection
- **Train-test split:** 80% training and 20% testing

The dataset is included in this repository.

## Technologies Used

- Python
- Pandas and NumPy
- Gensim (Word2Vec)
- Scikit-learn (Logistic Regression)
- Joblib

## Model Workflow

1. Load and preprocess the dataset.
2. Convert text into numerical vectors using pretrained Word2Vec embeddings.
3. Train a Logistic Regression classifier.
4. Evaluate the model on test data.
5. Predict whether new text is benign or a potential injection.

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 82.68% |
| Precision | 85.99% |
| Recall | 83.71% |
| F1-score | 84.83% |

## Project Structure

```text
Prompt-Injection-Detection-Model/
├── Docs/
├── models/
│   └── prompt_injection_model.joblib
├── dataset.csv
├── main.py
├── evaluate_model.py
├── prediction.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/AGO-Epr/Prompt-Injection-Detection-Model.git
cd Prompt-Injection-Detection-Model
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Train the model:

```bash
python main.py
```

Evaluate the model:

```bash
python evaluate_model.py
```

Run predictions:

```bash
python prediction.py
```

## Word2Vec Model

The pretrained Word2Vec files are not included in this repository because of their large size. You must download or provide the pretrained embeddings before running the scripts, depending on how the code is configured.

## Limitations

This model can make incorrect predictions. It is an experimental detection tool and should not be considered a complete security solution.

