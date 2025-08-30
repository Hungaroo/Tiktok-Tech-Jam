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
   - Produces `policy_final_fast` predictions quickly  

5. **Hugging Face Ensemble (Optional)**  
   - Zero-shot classification (`facebook/bart-large-mnli` or lite `distilbert`)  
   - Few-shot generation (`Qwen` models)  
   - Optional fine-tuned classifier  
   - HF-based sentiment classification  

6. **Evaluation & Reporting**  
   - Precision, recall, F1-score (if `policy_gt` / `sentiment_gt` columns exist)  
   - Confusion matrix (CSV + PNG)  
   - JSON summary of dataset metrics  

---

## Installation Guide

### Prerequisites
- Python 3.9+  
- (Optional) OpenAI API Key if using GPT-based sentiment  

### Install Requirements
```
pip install --upgrade pip
pip install -r requirements.txt
```


## Step to Run Codes Locally
```
python util.py --input reviews.csv --outdir outputs
```

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
