"""
model_comparison.py
───────────────────
Loads all saved models and evaluates them on the validation set.
Produces a ranked comparison table.

Run AFTER train_models.py
"""

import pandas as pd
import pickle
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    recall_score,
    f1_score,
    precision_score,
)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
DATA_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
TARGET     = "departure_delayed"

# ─────────────────────────────────────────────────────────────
# LOAD VALIDATION DATA
# ─────────────────────────────────────────────────────────────
print("Loading validation data...")
valid = pd.read_parquet(DATA_DIR / "valid_selected.parquet")

X_valid = valid.drop(columns=[TARGET])
y_valid = valid[TARGET]

print(f"  Valid : {X_valid.shape}")

# ─────────────────────────────────────────────────────────────
# LOAD SCALER  (for Logistic Regression only)
# ─────────────────────────────────────────────────────────────
with open(MODELS_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_valid_scaled = scaler.transform(X_valid)

# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# (display_name, filename, use_scaling)
# ─────────────────────────────────────────────────────────────
model_registry = [
    ("Logistic Regression (baseline)", "logistic_regression", True),
    ("Random Forest",                  "random_forest",       False),
    ("XGBoost",                        "xgboost",             False),
    ("LightGBM",                       "lightgbm",            False),
    ("Gradient Boosting",              "gradient_boosting",   False),
]

# ─────────────────────────────────────────────────────────────
# EVALUATE LOOP
# ─────────────────────────────────────────────────────────────
results = []

for display_name, filename, use_scaling in model_registry:

    print(f"\n{'='*60}")
    print(f"  {display_name}")
    print(f"{'='*60}")

    model_path = MODELS_DIR / f"{filename}.pkl"
    if not model_path.exists():
        print(f"  ✗ Model file not found: {model_path}")
        print(f"    Run train_models.py first.")
        continue

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    Xva = X_valid_scaled if use_scaling else X_valid.values

    y_pred  = model.predict(Xva)
    y_proba = model.predict_proba(Xva)[:, 1]

    roc  = roc_auc_score(y_valid, y_proba)
    rec1 = recall_score(y_valid, y_pred)
    pre1 = precision_score(y_valid, y_pred)
    f1_1 = f1_score(y_valid, y_pred)
    cm   = confusion_matrix(y_valid, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Classification Report:")
    print(classification_report(y_valid, y_pred, digits=3))
    print(f"  ROC-AUC         : {roc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True  Negatives (correct on-time)  : {tn:>8,}")
    print(f"    False Positives (false alarm)       : {fp:>8,}")
    print(f"    False Negatives (missed delay) ← ✗ : {fn:>8,}")
    print(f"    True  Positives (caught delay) ← ✓ : {tp:>8,}")

    results.append({
        "Model"            : display_name,
        "ROC-AUC"          : round(roc,  4),
        "Recall (Del)"     : round(rec1, 4),
        "Precision (Del)"  : round(pre1, 4),
        "F1 (Del)"         : round(f1_1, 4),
        "Missed Delays"    : fn,
        "Caught Delays"    : tp,
    })

# ─────────────────────────────────────────────────────────────
# FINAL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  FINAL COMPARISON  (sorted by F1 — delayed class)")
print(f"{'='*60}")

results_df = (
    pd.DataFrame(results)
    .sort_values("F1 (Del)", ascending=False)
    .reset_index(drop=True)
)
results_df.index += 1  # rank from 1

print(results_df.to_string())

# Print the winner clearly
winner = results_df.iloc[0]["Model"]
print(f"  ✓ SELECTED FOR TUNING: {winner}")
print()