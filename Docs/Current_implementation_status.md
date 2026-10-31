# Prompt Injection Detection Research Project: Prompt-Injection Detection Baseline

> **Project status:** Early research prototype\
> **Current implementation:** Binary prompt-text classification using
> pretrained Word2Vec embeddings and Logistic Regression.

## 1. Overview

Prompt Injection Detection Research Project is an exploratory project
investigating how prompt-injection risks can be detected and, in a
future extension, monitored across AI-agent workflows.

The current code is a **text classifier baseline**. It reads labelled
prompts, converts each prompt into a fixed-length vector using
pretrained Word2Vec word embeddings, and trains a Logistic Regression
model to classify each prompt as either benign or prompt injection.

This version does **not** yet monitor multi-agent message handoffs,
track the origin of instructions, inspect tool calls, or enforce
security policies before actions execute. Those are possible future
research extensions, not implemented capabilities.

## 2. Current objectives

The current stage aims to:

-   establish a reproducible prompt-injection text-classification
    baseline;
-   measure how well averaged Word2Vec features separate benign and
    injection-labelled prompts;
-   report more informative metrics than accuracy alone;
-   identify limitations that should guide the next research stage.

## 3. Current pipeline

``` text
dataset.csv
    |
    v
Data validation and cleaning
    |
    v
Pretrained Google News Word2Vec
    |
    v
Tokenise each prompt and retrieve word vectors
    |
    v
Average recognised word vectors
    |
    v
300-dimensional prompt representation
    |
    v
Stratified train/test split
    |
    v
Logistic Regression
    |
    v
Evaluation metrics and saved model files
```

### Dataset format

Place `dataset.csv` in the same folder as `main_improved.py`. It must
contain:

  Column    Meaning
  --------- ------------------------------------------
  `text`    Prompt text
  `label`   `0` for benign; `1` for prompt injection

The script removes missing and empty rows, validates the labels, checks
for identical text with conflicting labels, and removes exact duplicate
prompts.

**Dataset responsibility:** The dataset's provenance, licensing,
labelling quality, class balance, and representativeness must be
documented separately. The script cannot establish that labels are
correct.

## 4. Review of the original implementation

The original `main.py` already had a useful baseline structure:

-   loaded a CSV containing `text` and `label`;
-   removed missing and empty prompts;
-   mapped labels to benign (`0`) and injection (`1`);
-   loaded pretrained `word2vec-google-news-300`;
-   represented each prompt by averaging its recognised word vectors;
-   trained a class-weighted Logistic Regression classifier;
-   used a stratified 80/20 train/test split;
-   printed accuracy and a classification report;
-   saved the classifier and Word2Vec vectors.

These are reasonable starting choices for a lightweight baseline.
However, the original script left several research and engineering
issues to address.

## 5. Improvements in `main_improved.py`

  -----------------------------------------------------------------------
  Area                    Original implementation Revised implementation
  ----------------------- ----------------------- -----------------------
  File paths              Depended on the         Paths are anchored to
                          terminal's working      the script directory
                          directory               

  Dataset schema          Assumed required        Checks for `text` and
                          columns existed         `label`

  Labels                  Checked binary values   Also checks duplicate
                                                  text with conflicting
                                                  labels

  Class counts            Relied on               Checks for at least two
                          stratification to fail  examples in each class
                          if unsuitable           

  Word2Vec                Averaged recognised     Preserves the same
                          word vectors            averaging method

  Vocabulary coverage     Not measured            Reports
                                                  recognised-token
                                                  coverage and
                                                  zero-coverage prompts

  Logistic Regression     `max_iter=1000`         `max_iter=2000`;
                                                  retains balanced class
                                                  weights

  Metrics                 Accuracy and            Adds precision, recall,
                          classification report   F1, ROC-AUC when
                                                  defined, and confusion
                                                  matrix

  Saved outputs           Classifier and vectors  Also writes evaluation
                                                  metrics to JSON

  Organisation            Mostly top-level script Uses functions for
                                                  loading, vectorisation,
                                                  and execution
  -----------------------------------------------------------------------

### Why keep Word2Vec averaging initially?

The purpose of this revision is to improve the reliability and
evaluation of the existing baseline, not to change several variables at
once.

Each recognised token is represented by a 300-dimensional pretrained
vector. The script averages those vectors to create one fixed-size
representation per prompt. This is compatible with the current Logistic
Regression classifier.

Averaging is simple and computationally manageable, but it discards word
order and much of the context in which words appear. It can also produce
weak representations when many prompt tokens are missing from the
Word2Vec vocabulary. Vocabulary coverage is therefore reported as a
diagnostic.

## 6. Evaluation: how to interpret the results

The revised script reports:

-   **Accuracy:** fraction of all test examples classified correctly.
-   **Precision (injection class):** among prompts predicted as
    injection, the fraction labelled injection.
-   **Recall (injection class):** among injection-labelled prompts, the
    fraction detected.
-   **F1-score:** harmonic mean of injection precision and recall.
-   **ROC-AUC:** ranking performance across classification thresholds,
    when both classes are present in the test set.
-   **Confusion matrix:** counts of true benign, false injection alarms,
    missed injections, and correctly detected injections.

For a security-oriented detector, do not report accuracy alone. Discuss
false positives and false negatives, and explain the operational
trade-off between them.

The script's 80/20 random stratified split is an initial baseline only.
Exact duplicate removal does not prevent near-duplicate or
shared-template leakage. If related prompts appear in both training and
test sets, measured performance may be optimistic.

No performance result should be claimed until the script has been run on
the actual dataset and the resulting metrics have been recorded.

## 7. Known limitations

1.  **Loss of context:** Averaging word embeddings ignores word order
    and sentence-level meaning.
2.  **Vocabulary gaps:** Out-of-vocabulary words do not contribute to
    the prompt vector. An all-zero vector contains no useful lexical
    signal.
3.  **Dataset dependence:** Results depend on the dataset's source,
    labelling rules, balance, and diversity.
4.  **Potential split leakage:** Random splitting may put paraphrases or
    examples from the same template in both sets.
5.  **Distribution shift:** Performance on one dataset does not
    establish performance on prompts from different systems, domains, or
    attack families.
6.  **No agent provenance:** The implementation does not record which
    agent or source introduced an instruction or how it was transformed
    during handoffs.
7.  **No action enforcement:** It does not inspect or block tool calls
    or other consequential agent actions.
8.  **No guarantee of safety:** A binary classifier can make mistakes
    and must not be treated as a complete security boundary.

## 8. Novelty and research contribution

### What is not novel by itself

The following should not be presented as the project's research novelty:

-   using Word2Vec to represent text;
-   averaging word embeddings into sentence or prompt vectors;
-   using Logistic Regression for binary text classification;
-   building a standalone prompt-injection detector;
-   reporting standard classification metrics.

These are established techniques. A working implementation is useful
engineering, but does not by itself establish a new research
contribution.

### A more defensible direction for novelty

A stronger research direction is to investigate **semantic influence and
provenance across multi-agent handoffs, together with deterministic
policy checks before proposed tool actions are executed**.

A possible research question is:

> How robustly can semantic-provenance tracking and pre-execution policy
> checks preserve correct security decisions across nondeterministic
> multi-agent runs, particularly when untrusted content is paraphrased
> and benign tasks discuss that same content?

This is a proposed direction, not a claim that the current code already
achieves it.

Potential research contributions, if implemented and experimentally
validated, could include:

1.  **Provenance tracking:** record the source of content and its path
    through agent messages, summaries, and handoffs.
2.  **Semantic influence tracking:** investigate whether untrusted
    instructions continue to influence later agent messages, even after
    paraphrasing or summarisation.
3.  **Pre-execution policy enforcement:** evaluate proposed actions
    against explicit policies before execution, rather than relying only
    on a prompt-text classification result.
4.  **Reproducible decision logging:** record inputs, provenance, policy
    version, proposed action, decision, and reason so that the same
    recorded case can be audited.
5.  **Robustness evaluation:** test paraphrased untrusted instructions,
    multi-agent transformations, benign discussion of suspicious text,
    and distribution shifts.

### How to position novelty responsibly

Related research already explores prompt-injection detection,
provenance, information-flow tracking, and agent/tool safeguards. In
particular, work such as CaMeL, Agent-Sentry, TrustTrace, and SafeFlow
is relevant to the proposed direction. SafeFlow appears especially close
to the broad idea of propagating semantic risk across agent
collaboration and checking actions.

Therefore:

-   do not claim that provenance-aware multi-agent prompt-injection
    defence is entirely new;
-   compare the eventual system against the closest prior work;
-   identify a narrow, testable difference in the method or evaluation;
-   support any novelty claim with a current literature review and
    experimental evidence.

A potentially defensible contribution may be a carefully specified
evaluation of how provenance and policy decisions behave under
paraphrase, benign discussion, agent handoffs, and repeated runs---not
simply the combination of a classifier and a logging system.

## 9. Recommended research roadmap

### Stage 1 --- Establish the baseline

-   Run `main_improved.py` on the documented dataset.
-   Record dataset sources, class counts, and labelling rules.
-   Save the full classification report and confusion matrix.
-   Inspect examples of false positives and false negatives.

### Stage 2 --- Strengthen the evaluation

-   Create a source-based or template-grouped test split to reduce
    leakage.
-   Add a separate set of benign prompts that discuss suspicious
    instructions without following them.
-   Evaluate paraphrase robustness and out-of-distribution prompts.
-   Compare against a TF-IDF + Logistic Regression baseline.
-   Report per-class metrics and uncertainty where feasible.

### Stage 3 --- Investigate agent-level provenance

-   Define what counts as trusted and untrusted content.
-   Define how provenance is attached to messages and preserved across
    handoffs.
-   Record transformations such as quoting, summarising, and
    paraphrasing.
-   Keep provenance evidence separate from the model's predicted class.

### Stage 4 --- Add policy checks before action execution

-   Define a small, explicit set of tool-action policies.
-   Evaluate proposed actions before execution.
-   Log the proposed action, relevant provenance, policy version,
    decision, and reason.
-   Test whether the enforcement layer behaves consistently on the same
    recorded inputs and policy.

Deterministic enforcement means that the policy decision is reproducible
for the same recorded inputs and policy. It does **not** mean that the
language model itself becomes deterministic.

### Stage 5 --- Compare with related work

-   Select the closest comparable systems and benchmarks.
-   Compare threat models, assumptions, capabilities, and limitations.
-   Reproduce comparable tests where possible.
-   State clearly which parts are implemented, evaluated, or only
    proposed.

## 10. Running the project on Windows with VS Code

### Requirements

Use a Python environment compatible with the installed versions of
NumPy, Gensim, and scikit-learn. Python 3.11 is a reasonable starting
point.

### Folder layout

``` text
Prompt Injection Detection Research Project/
├── main_improved.py
├── dataset.csv
└── models/                 # created by the script
```

### Create and activate a virtual environment

In the VS Code terminal:

``` powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, select Command Prompt as the terminal
profile and run:

``` cmd
.venv\Scripts\activate.bat
```

### Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install numpy pandas gensim scikit-learn joblib
```

### Run

``` powershell
python main_improved.py
```

The first run downloads the pretrained Google News Word2Vec vectors and
requires an internet connection. The model is large and may take time
and disk space to download.

### Generated files

After a successful run:

``` text
models/
├── prompt_injection_model.joblib
├── word2vec_model.kv
└── evaluation_metrics.json
```

The classifier and vector file must be used with the same tokenisation
and vectorisation procedure during inference. The current script is a
training/evaluation script; a separate inference interface should be
added and tested before integrating it into another application.

## 11. Reproducibility notes

The script fixes the train/test split seed and configures Logistic
Regression with a random state. This helps reproduce the same split and
model configuration under the same data and software environment.

For stronger reproducibility, also record:

-   dataset version, source, and checksum;
-   Python and package versions;
-   Word2Vec model identifier;
-   exact code revision;
-   split strategy and random seed;
-   all evaluation outputs.

The pretrained Word2Vec model is downloaded separately. Its availability
and compatibility should be verified in the environment used for the
experiment.

## 12. References for the novelty review

These works are starting points for a literature review, not proof that
the proposed system is novel or that it has been experimentally compared
with them:

-   **CaMeL:** https://arxiv.org/abs/2503.18813
-   **Agent-Sentry:** https://arxiv.org/abs/2603.22868
-   **SafeFlow:** https://arxiv.org/abs/2607.25255
-   **TrustTrace:**
    https://ieeexplore.ieee.org/abstract/document/11664419/
-   **PIDS-Bench:**
    https://github.com/ShirePyDev/Prompt-Injection-Detection-System

Verify publication status, dates, versions, and the exact claims of each
paper before citing them in a final academic submission. Search for
additional recent work and compare the actual methods rather than
relying on titles or abstracts alone.

------------------------------------------------------------------------

## Project status summary

  Capability                                       Status
  ------------------------------------------------ -------------------------------
  Prompt text classification                       Implemented in baseline
  Averaged Word2Vec representation                 Implemented
  Logistic Regression                              Implemented
  Basic test-set evaluation                        Implemented
  Vocabulary coverage diagnostic                   Implemented in revised script
  Cross-source / template-held-out evaluation      Not yet implemented
  Robustness and benign-discussion evaluation      Not yet implemented
  Multi-agent provenance tracking                  Proposed
  Semantic influence tracking across handoffs      Proposed
  Pre-execution policy enforcement                 Proposed
  Comparative evaluation against related systems   Not yet completed

**Current claim:** Prompt Injection Detection Research Project is an
early prompt-injection classification baseline and a foundation for
investigating provenance-aware safeguards in multi-agent systems. Its
broader research novelty remains to be established through
implementation, comparison with prior work, and rigorous evaluation.
