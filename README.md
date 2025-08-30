# Tiktok-Tech-Jam

### Filtering the Noise: ML for Trustworthy Location Reviews

---

## Table of Contents
- [Team Members](#team-members)
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation Guide](#installation-guide)
  - [Prerequisites](#prerequisites)
  - [Install Requirements](#install-requirements)
- [Step to Run Codes Locally](#step-to-run-codes-locally)
- [Pipeline Options](#pipeline-options)
- [Evaluation & Reporting](#evaluation--reporting)
  - [Summary of findings and recommendations](#summary-of-findings-and-recommendations)
  - [What was evaluated (scope)](#what-was-evaluated-scope)
  - [Overall metrics (proxy: model vs rule labels)](#overall-metrics-proxy-model-vs-rule-labels)
  - [Class-level signals (for debugging only)](#class-level-signals-for-debugging-only)
  - [How to reproduce this exact evaluation](#how-to-reproduce-this-exact-evaluation)
  - [Recommendations](#recommendations)
- [Tech Stack](#tech-stack)

---

## Team Members
| Name              |
|-------------------|
| Cheryl Toh Wen Qi |
| Chua Xinhui       |
| Chua Xinyan       |
| Lee Jia Wen       |

---

## Project Overview
Online reviews significantly influence user perception of local businesses, but irrelevant, misleading, or spammy reviews reduce trust.  

This project develops an **end-to-end ML pipeline** to automatically evaluate review **quality**, **relevancy**, and **sentiment**, enforcing policy categories such as **advertisement**, **irrelevant content**, **rant without visit**, and **clean**.  

We provide three complementary pipelines:  
1. **Base pipeline**: preprocessing, feature engineering, sentiment (OpenAI/HF/lexicon), and rule-based policy detection  
2. **Fast classifier**: TF-IDF + LinearSVC (trained on weak labels)  
3. **Hugging Face Ensemble (optional)**: zero-shot NLI, few-shot LLM, optional fine-tuned classifier, and ensemble policy prediction  

The system balances **speed**, **accuracy**, and **scalability** to improve the reliability of location-based reviews.

---

## Features
1. **Preprocessing & Cleaning**  
   - Removes URLs, HTML, emojis, duplicates  
   - Standardizes text and metadata  

2. **Feature Engineering**  
   - Extracts word/char counts, avg word length, keywords  
   - Captures posting time, GPS proximity, user metadata  

3. **Policy Enforcement**  
   - Rule-based detection of ads, irrelevancy, rants-without-visit, spam, malicious links, PII, and self-promotion  

4. **Fast Classifier (TF-IDF + LinearSVC)**  
   - Learns from weak rule-based labels  
   - Produces `policy_fast_ml` (trained model) and `policy_final_fast` (simple ensemble) predictions quickly  

5. **Hugging Face Ensemble (Optional)**  
   - Zero-shot classification (`facebook/bart-large-mnli` or lite `distilbert`)  
   - Few-shot generation (`Qwen` models)  
   - Optional fine-tuned classifier  
   - HF-based sentiment classification  

6. **Evaluation & Reporting**  
   - Precision, recall, F1-score (if `policy_gt` / `sentiment_gt` columns exist)  
   - Confusion matrix PNG (generated when ground-truth labels are provided)  
   - JSON summary of dataset metrics

---

## Installation Guide

### Prerequisites
- Python 3.9+  
- (Optional) OpenAI API Key if using GPT-based sentiment  

### Install Requirements
```bash
pip install --upgrade pip
```
```
pip install -r requirements.txt
```

## Step to Run Codes Locally
```bash
python util.py --input reviews.csv --outdir outputs --run_base false --run_hf false --run_fast_clf true --fast_min_df 5 --fast_max_df 0.9
```
## Evaluation & Reporting 
### Summary of findings and recommendations
This section documents **exactly what was run and measured** in the latest evaluation artifact in this repo.

### What was evaluated (scope)
- **Model path:** the **Fast Classifier** (TF-IDF + LinearSVC) trained on **rule-based weak labels**.
- **Evaluation type:** **proxy evaluation** — the trained model’s predictions were compared **against the rule-based labels**, not human ground truth (no `policy_gt` column was present in this run).
- **Dataset size:** `1,100` rows.
- **Classes encountered:** `advertisement`, `clean`, `rant_without_visit`, `self_promotion`, `spam_or_lowinfo`.
- **Training status for this run:** `model_trained: false` (an existing fitted model was used; no re-training occurred during this run).

### Overall metrics (proxy: model vs rule labels)
These are **overall** (not per-class) metrics reported by the run:

- **Accuracy:** `0.9845454545`  
- **Macro Precision:** `0.9867036525`  
- **Macro Recall:** `0.9326535790`  
- **Macro F1:** `0.9571548543`

(Weighted averages for reference: Precision `0.9843108812`, Recall `0.9845454545`, F1 `0.9838502462`.)

> Because this is a **proxy** check (model vs rules), treat these as alignment with the rule system, **not** true real-world accuracy.

### Class-level signals (for debugging only)
While our reporting focuses on overall metrics, the run also logged per-class stats that explain where the proxy disagreement is concentrated:

| class               | precision | recall  | f1       | support |
|---------------------|-----------|---------|----------|---------|
| advertisement       | 1.0000    | 0.8824  | 0.9375   | 17      |
| clean               | 0.9845    | 0.9976  | 0.9910   | 827     |
| rant_without_visit  | 1.0000    | 1.0000  | 1.0000   | 1       |
| self_promotion      | 0.9898    | 1.0000  | 0.9949   | 195     |
| spam_or_lowinfo     | 0.9592    | 0.7833  | 0.8624   | 60      |

> The **lowest proxy recall** is in `spam_or_lowinfo` (≈ 0.78), followed by `advertisement` (≈ 0.88). This indicates the trained model most often diverges from the rule system on these two categories.
> 
### Recommendations
- **Switch from proxy to true evaluation.** Add a human ground-truth column (e.g., `policy_gt`) and re-run to measure **actual** model performance rather than rule alignment.
- **Retrain the fast classifier on this dataset snapshot.** The report shows `model_trained: false`; re-run with `--retrain_fast` so results reflect the latest data distribution.
- **Prioritize recall improvements where disagreement is highest.** Focus on `spam_or_lowinfo` (recall ≈ 0.7833) and `advertisement` (recall ≈ 0.8824) first.
- **Address class imbalance before the next training pass.** `clean` dominates (827 items) while `advertisement` (17) and `rant_without_visit` (1) are scarce; collect more labeled samples or use class weighting/sampling.

---

## Tech Stack

#### Core Language
- [![Python](https://img.shields.io/badge/Python%203.9+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) - Main language for data pipelines and orchestration.

#### Data Processing
- [![pandas](https://img.shields.io/badge/pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/) – CSV reading, dataframe manipulation, saving outputs  
- [![NumPy](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/) – numerical features (lengths, averages, vector ops)

#### Machine Learning
- [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) – TF-IDF vectorizer + LinearSVC (fast supervised classifier)  
- [![joblib](https://img.shields.io/badge/joblib-6DB33F.svg?style=for-the-badge&logo=python&logoColor=white)](https://joblib.readthedocs.io/) – model persistence (save/load trained classifier)

#### NLP & Transformers (Optional HF Path)
- [![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/) – zero-shot, few-shot, and optional fine-tuned classification  
- [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) – deep learning backend required by Hugging Face models  
- [![langdetect](https://img.shields.io/badge/langdetect-4B8BBE.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/langdetect/) – lightweight language detection for rule enforcement

#### Visualization & Reporting
- [![matplotlib](https://img.shields.io/badge/matplotlib-0C55A5.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://matplotlib.org/) – confusion matrices (PNG)  
- [![scikit-learn](https://img.shields.io/badge/sklearn%20metrics-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) – `classification_report`, `confusion_matrix`

#### Backend & Orchestration
- [![CLI](https://img.shields.io/badge/CLI-181717.svg?style=for-the-badge&logo=windowsterminal&logoColor=white)](./util.py) – orchestrates base pipeline, HF ensemble, and fast classifier  
- [![JSON](https://img.shields.io/badge/JSON-000000.svg?style=for-the-badge&logo=json&logoColor=white)](https://www.json.org/) + [![CSV](https://img.shields.io/badge/CSV-217346.svg?style=for-the-badge&logo=microsoft-excel&logoColor=white)](https://en.wikipedia.org/wiki/Comma-separated_values) – structured reporting and downstream outputs
