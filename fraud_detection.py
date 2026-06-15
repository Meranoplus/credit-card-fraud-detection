import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score, recall_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegressionCV
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

rf_params = {
    'n_estimators': 300,
    'random_state': 42,
    'n_jobs': -1
}

xgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': 42,
    'n_jobs': -1
}

lgbm_params = {
    'n_estimators': 500,
    'learning_rate': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'objective': 'binary',
    'metric': 'average_precision'
}

cat_params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 6,
    'random_state': 42,
    'verbose': 0
}

lr_params = {
    'max_iter': 1000,
    'cv': 5,               
    'random_state': 42,
    'n_jobs': -1
}

# ── 1. LOAD DATA ─────────────────────────────────
df = pd.read_csv("creditcard.csv")
X = df.drop('Class', axis=1)
y = df['Class']

# ── 2. SPLIT ─────────────────────────────────────
# 50% train, 25% val, 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)
X_val, X_test, y_val, y_test     = train_test_split(X_test, y_test, random_state=42, test_size=0.5)

# ── 3. SCALE ─────────────────────────────────────
# fit only on train to avoid data leakage
scaler = StandardScaler()
X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_val[["Time", "Amount"]]   = scaler.transform(X_val[["Time", "Amount"]])
X_test[["Time", "Amount"]]  = scaler.transform(X_test[["Time", "Amount"]])

# ── 4. BALANCE ───────────────────────────────────
# SMOTE only on training data — never on val/test
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

models = { 
    "rf": RandomForestClassifier(**rf_params),
    "logistic": LogisticRegressionCV(**lr_params),
    "Xgb": XGBClassifier(**xgb_params),
    "cat": CatBoostClassifier(**cat_params),
    # "Lgbm": LGBMClassifier(**lgbm_params)
}

# ── 5. TRAIN & EVALUATE ──────────────────────────
# PR-AUC is primary metric — more reliable than ROC-AUC on imbalanced data
results = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
        "PR-AUC":    average_precision_score(y_test, y_proba)
    }

results_df = pd.DataFrame(results).T
print(results_df.sort_values("PR-AUC", ascending=False))
