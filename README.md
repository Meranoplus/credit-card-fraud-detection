# Credit Card Fraud Detection

Comparing 5 machine learning models on an extremely imbalanced dataset (0.2% fraud rate).

## Problem

Standard accuracy is misleading on imbalanced data.
A model predicting everything as legit gets 99.8% accuracy but catches zero fraud.
Primary metric used: **PR-AUC** (Precision-Recall AUC), since ROC-AUC is overly optimistic when the positive class is this rare.

## What I Did

- 80/10/10 train/val/test split
- StandardScaler fit on train only, applied to Time and Amount
- SMOTE applied only on training data to handle class imbalance — val/test are left untouched to stay realistic
- Compared 5 models: Random Forest, XGBoost, LightGBM, CatBoost, Logistic Regression
- Model selection done on the validation set only; test set evaluated once, at the end, for an unbiased final read

## Results (test set)

| Model          | Precision | Recall | F1   | PR-AUC |
|----------------|-----------|--------|------|--------|
| Random Forest  | 0.85      | 0.81   | 0.83 | 0.87   |
| XGBoost        | 0.77      | 0.84   | 0.80 | 0.87   |
| CatBoost       | 0.55      | 0.86   | 0.67 | 0.84   |
| LightGBM       | 0.77      | 0.84   | 0.80 | 0.79   |
| Logistic Reg   | 0.05      | 0.91   | 0.10 | 0.77   |

**Winner: XGBoost**, selected on the validation set (PR-AUC 0.904, narrowly ahead of Random Forest's 0.903). On the held-out test set, Random Forest edges ahead instead (0.867 vs 0.865) — but the two models are statistically tied; which one ranks first depends on the specific data split. Both are reported above for transparency, with the original val-based selection (XGBoost) kept as the locked-in answer rather than re-picking based on test.

## Key Findings

- Tree-based and boosted models significantly outperform Logistic Regression — confirms fraud patterns are non-linear.
- **Random Forest and XGBoost are essentially tied.** Across multiple validation/test splits, the two models swap first place by margins of 0.002–0.006 PR-AUC. Neither is reliably "better" — pick either with confidence.
- **CatBoost has a real, reproducible precision problem.** Recall stays strong (~0.86), but precision repeatedly lands around 0.52–0.56 across every split tested. Inspecting the probability distribution for true fraud cases showed scores clustering near 0 or 1, not in the adjustable middle — so the issue isn't a misplaced default threshold, it's the model's confidence calibration itself.
- **Threshold tuning was tested and didn't help.** Given the above, lowering or raising the 0.5 cutoff had little effect, since there were few borderline-probability cases to move across the boundary. This is a case where the standard fix didn't apply — worth knowing rather than assuming threshold tuning is always the answer.
- Logistic Regression's precision collapses to ~0.05–0.07 despite high recall, confirming a linear decision boundary is a poor fit for this problem.

## Stack

Python, Scikit-learn, XGBoost, LightGBM, CatBoost, Imbalanced-learn

## Dataset

This repo does not include the dataset due to file size.

**Download:** [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Setup:**
1. Download `creditcard.csv` from the Kaggle link above (requires a free Kaggle account).
2. Place `creditcard.csv` in the same folder as `fraud_detection.py`.
3. Install dependencies:
   ```
   pip install pandas scikit-learn imbalanced-learn catboost xgboost lightgbm
   ```
4. Run the script:
   ```
   python fraud_detection.py
   ```

The script will print validation results, the selected winning model, and final test results to the console.
