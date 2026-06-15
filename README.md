# Credit Card Fraud Detection

Comparing 5 machine learning models on an extremely imbalanced dataset (0.2% fraud rate).

## Problem
Standard accuracy is misleading on imbalanced data.
A model predicting everything as legit gets 99.8% accuracy but catches zero fraud.
Primary metric used: **PR-AUC** (Precision-Recall AUC).

## What I Did
- Train/Val/Test split to prevent data leakage
- StandardScaler on Time and Amount only
- SMOTE applied only on training data to handle class imbalance
- Compared 5 models: Random Forest, XGBoost, LightGBM, CatBoost, Logistic Regression

## Results
| Model          | Precision | Recall | F1   | PR-AUC |
|----------------|-----------|--------|------|--------|
| Random Forest  | 0.88      | 0.83   | 0.86 | 0.86   |
| XGBoost        | 0.75      | 0.85   | 0.80 | 0.85   |
| CatBoost       | 0.52      | 0.87   | 0.65 | 0.83   |
| Logistic Reg   | 0.06      | 0.88   | 0.11 | 0.72   |

**Winner: Random Forest** with best F1 (0.86) and PR-AUC (0.86)

## Key Findings
- Tree-based models significantly outperform Logistic Regression
- Confirms fraud patterns are non-linear
- LightGBM excluded — probability calibration issues on this dataset

## Stack
Python, Scikit-learn, XGBoost, LightGBM, CatBoost, Imbalanced-learn

## Dataset
Not included due to file size.
Download creditcard.csv from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it in the same folder as fraud_detection.py before running.
