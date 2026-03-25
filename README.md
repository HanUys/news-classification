# News Article Topic Classification

## Overview

This project implements a multi-class machine learning pipeline for classifying news articles into seven categories using both textual content and structured metadata. The objective is to build a robust and efficient model that generalizes well to future data while handling class imbalance and high-dimensional sparse features.

---

## Dataset

The dataset consists of 99,997 news articles divided into two subsets:

- Development set: 79,997 samples (used for training and validation)
- Evaluation set: 20,000 samples (used for final prediction)

Each sample includes the following features:
- Title (text)
- Article body (text)
- Source (publisher)
- Page rank (numerical signal)
- Timestamp (publication time)

The dataset exhibits class imbalance across seven categories, making Macro-F1 the primary evaluation metric.

---

## Methodology

### Preprocessing

- Concatenated title and article into a single text input
- Applied light cleaning to remove only non-informative artifacts
- Preserved useful textual patterns to retain discriminative signals
- Sorted data chronologically to enable time-based validation

### Feature Engineering

Text features:
- TF-IDF vectorization
- Word n-grams (1 to 3)
- Vocabulary size up to 30,000 features

Metadata features:
- Source encoding (top 250 publishers, others grouped as "Other")
- Page rank normalized using Min-Max scaling
- Temporal features:
  - Day segment
  - Day of week
  - Weekend indicator
  - Month index

The final representation is a high-dimensional sparse feature matrix (~25,000+ features).

---

## Model Selection

Two linear models were evaluated:

- Linear Support Vector Machine (LinearSVC)
- Logistic Regression

Linear models were preferred due to their efficiency and strong performance on sparse TF-IDF representations.

---

## Hyperparameter Tuning

- Evaluation metric: Macro-F1
- Validation strategy: Chronological split (70% training, 30% validation)
- Grid search over regularization parameter (C) and model configurations

Best performing model:

LinearSVC(
C=0.1,
class_weight="balanced",
dual=False,
max_iter=2000
)


---

## Results

| Model               | C    | Macro-F1 | Training Time |
|--------------------|------|----------|--------------|
| LinearSVC          | 0.1  | 0.7438   | 16.98 s      |
| Logistic Regression| 1.0  | 0.7394   | 31.48 s      |

- Overall accuracy: approximately 75%
- Strong performance on well-defined classes such as Sports and Tech/Science
- Lower performance on General News due to semantic overlap with other categories

---

## Key Challenges

- Class imbalance across categories
- Semantic overlap between similar classes (e.g., General vs International)
- Long-tail distribution of publishers

---

## Key Insights

- Light text cleaning outperformed aggressive preprocessing by preserving useful patterns
- Metadata features significantly improved model stability
- Proper regularization (C = 0.1) balanced bias and variance effectively
- Chronological validation prevented temporal data leakage

---

## Project Structure
News-Project/
│
├── data/
├── notebooks/
├── src/
├── requirements.txt
├── README.md
└── .venv/


---

## Setup Instructions

1. Create a virtual environment:
python -m venv .venv

2. Activate the environment:
.venv\Scripts\Activate

3. Install dependencies:
pip install -r requirements.txt


---

## Reproducibility

- Fixed random seed (random_state=42)
- All preprocessing steps fitted on training data only
- Chronological split ensures realistic evaluation without leakage

---

## Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy
- TF-IDF Vectorization
- LinearSVC, Logistic Regression

---

## Summary

This project demonstrates the design of a scalable and reliable machine learning pipeline for text classification, combining natural language processing with structured data features. It highlights practical solutions to real-world challenges such as class imbalance, temporal validation, and high-dimensional sparse modeling.