# Prompt Injection Detection Model

A deep learning-based system that detects potential prompt injection attacks using a fine-tuned DistilBERT transformer model.

## Overview

Prompt injection is a security concern in AI applications where specially crafted inputs attempt to manipulate a language model's behavior or override its instructions.

This project uses Natural Language Processing (NLP) and transfer learning to classify input prompts into two categories:

- **Benign Prompt (0):** Normal user instructions or queries.
- **Potential Prompt Injection (1):** Prompts identified as potentially containing injection attempts.

The model is built using DistilBERT and fine-tuned on a labeled prompt injection dataset.

## Model Architecture

- **Base Model:** DistilBERT (`distilbert-base-uncased`)
- **Framework:** PyTorch
- **Library:** Hugging Face Transformers
- **Task:** Binary Text Classification
- **Maximum Sequence Length:** 256 tokens
- **Optimizer:** AdamW

## Dataset

The dataset contains approximately 29,019 labeled text samples.

| Label | Category |
|---|---|
| 0 | Benign Prompt |
| 1 | Potential Prompt Injection |

The dataset is divided into training, validation, and testing sets using stratified sampling.

- Training: 81%
- Validation: 9%
- Testing: 10%

## Model Performance

The fine-tuned DistilBERT model achieved the following results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 91.94% |
| Precision | 91.07% |
| Recall | 95.41% |
| F1-Score | 93.19% |

These results represent performance on the project's test dataset and do not guarantee detection of every prompt injection technique.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Pandas
- NumPy
- Scikit-learn
- Git LFS

## Project Structure

```text
Prompt-Injection-Detection-Model/
│
├── Docs/
│
├── models/
│   └── distilbert_final/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
│
├── data/
│
├── dataset.csv
├── train_final_distilbert.py
├── prediction.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/AGO-Epr/Prompt-Injection-Detection-Model.git
```

Navigate to the project directory:

```bash
cd Prompt-Injection-Detection-Model
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For NVIDIA GPU support, install the appropriate CUDA-enabled PyTorch version. Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Model Usage

The fine-tuned model is stored in:

```text
models/distilbert_final/
```

Run the prediction script:

```bash
python prediction.py
```

Enter a prompt when requested. The model returns:

- Predicted category
- Class label
- Confidence score

Type `exit` to stop the program.

## Limitations

- The model is trained on a specific labeled dataset and may not recognize every possible prompt injection technique.
- Predictions depend on the quality and diversity of the training data.
- Confidence scores are model outputs and are not necessarily calibrated probabilities.
- This classifier should be used as one layer of a broader AI security system, not as a complete security solution.

## Future Improvements

- Expand the dataset with more diverse prompt injection examples.
- Evaluate performance on unseen attack patterns.
- Improve model robustness and generalization.
- Explore integration with LLM-based applications.

## Disclaimer

This project is developed for educational and research purposes to explore prompt injection detection using deep learning and transformer-based NLP models.
