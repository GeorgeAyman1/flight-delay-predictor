"""
model_comparison.py  — Enhanced
─────────────────────────────────
Loads all saved models and evaluates them on the validation set.

Key changes vs v1:
  • Evaluates at BOTH default threshold (0.5) AND optimal F1 threshold
    (saved by model_training.py) — shows the true ceiling of each model
  • Added PR-AUC (Precision-Recall AUC) metric — more informative than
    ROC-AUC for imbalanced classes (EDA confirmed 3.4:1 ratio)
  • Covers new models: extra_trees, hist_gradient_boosting, catboost
  • Final table sorted by F1 at optimal threshold
  • Cleaner selection rationale printed at the end

Run AFTER model_training.py
"""

import pandas as pd
import pickle
import numpy as np
import json
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,   # PR-AUC
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
# LOAD SCALER & THRESHOLDS
# ─────────────────────────────────────────────────────────────
with open(MODELS_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
X_valid_scaled = scaler.transform(X_valid)

thresholds_path = MODELS_DIR / "thresholds.json"
if thresholds_path.exists():
    with open(thresholds_path) as f:
        threshold_map = json.load(f)
else:
    threshold_map = {}
    print("  [warn] thresholds.json not found — using 0.5 for all models")

# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# (display_name, filename, use_scaling)
# ─────────────────────────────────────────────────────────────
model_registry = [
    ("Logistic Regression (baseline)", "logistic_regression",      True),
    ("Random Forest",                  "random_forest",             False),
    ("Extra Trees",                    "extra_trees",               False),
    ("XGBoost",                        "xgboost",                   False),
    ("LightGBM",                       "lightgbm",                  False),
    ("Hist Gradient Boosting",         "hist_gradient_boosting",    False),
    ("Gradient Boosting",              "gradient_boosting",         False),
    ("CatBoost",                       "catboost",                  False),
]

# ─────────────────────────────────────────────────────────────
# EVALUATE LOOP
# ─────────────────────────────────────────────────────────────
results = []

for display_name, filename, use_scaling in model_registry:

    model_path = MODELS_DIR / f"{filename}.pkl"
    if not model_path.exists():
        print(f"\n  [skip] {display_name} — model file not found")
        continue

    print(f"\n{'='*60}")
    print(f"  {display_name}")
    print(f"{'='*60}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    Xva     = X_valid_scaled if use_scaling else X_valid.values
    y_proba = model.predict_proba(Xva)[:, 1]

    # ── default threshold (0.5) ──────────────────────────────
    y_pred_def = (y_proba >= 0.5).astype(int)

    # ── optimal F1 threshold ─────────────────────────────────
    thresh     = threshold_map.get(filename, 0.5)
    y_pred_opt = (y_proba >= thresh).astype(int)

    # ── metrics ──────────────────────────────────────────────
    roc_auc  = roc_auc_score(y_valid, y_proba)
    pr_auc   = average_precision_score(y_valid, y_proba)   # NEW

    # at default threshold
    f1_def   = f1_score(y_valid, y_pred_def)
    rec_def  = recall_score(y_valid, y_pred_def)
    pre_def  = precision_score(y_valid, y_pred_def)
    cm_def   = confusion_matrix(y_valid, y_pred_def)
    tn_d, fp_d, fn_d, tp_d = cm_def.ravel()

    # at optimal threshold
    f1_opt   = f1_score(y_valid, y_pred_opt)
    rec_opt  = recall_score(y_valid, y_pred_opt)
    pre_opt  = precision_score(y_valid, y_pred_opt)
    cm_opt   = confusion_matrix(y_valid, y_pred_opt)
    tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()

    print(f"\n  ── At default threshold (0.50) ──")
    print(classification_report(y_valid, y_pred_def, digits=3))

    print(f"  ── At optimal threshold ({thresh:.3f}) ──")
    print(classification_report(y_valid, y_pred_opt, digits=3))

    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}   ← NEW: better metric for imbalanced classes")

    print(f"\n  Confusion Matrix @ optimal threshold:")
    print(f"    True  Negatives (correct on-time)  : {tn_o:>8,}")
    print(f"    False Positives (false alarm)       : {fp_o:>8,}")
    print(f"    False Negatives (missed delay) ← ✗ : {fn_o:>8,}")
    print(f"    True  Positives (caught delay) ← ✓ : {tp_o:>8,}")

    results.append({
        "Model"              : display_name,
        "ROC-AUC"            : round(roc_auc, 4),
        "PR-AUC"             : round(pr_auc,  4),
        "Threshold"          : round(thresh,  3),
        # optimal threshold metrics
        "Recall (opt)"       : round(rec_opt, 4),
        "Precision (opt)"    : round(pre_opt, 4),
        "F1 (opt)"           : round(f1_opt,  4),
        # default threshold metrics
        "F1 (0.5)"           : round(f1_def,  4),
        "Missed Delays"      : fn_o,
        "Caught Delays"      : tp_o,
    })

# ─────────────────────────────────────────────────────────────
# FINAL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  FINAL COMPARISON  (sorted by F1 at optimal threshold)")
print(f"{'='*60}")

results_df = (
    pd.DataFrame(results)
    .sort_values("F1 (opt)", ascending=False)
    .reset_index(drop=True)
)
results_df.index += 1

print(results_df.to_string())

print(f"""
─────────────────────────────────────────────────────────────
HOW TO READ THIS TABLE
─────────────────────────────────────────────────────────────
  ROC-AUC      → Overall separation ability (threshold-free)
  PR-AUC       → Better metric for imbalanced classes:
                 measures area under Precision-Recall curve
  Threshold    → Optimal cut-off found during training
  Recall (opt) → % of real delays caught at optimal threshold
  F1 (opt)     → Best achievable F1 for each model
  F1 (0.5)     → F1 at naive 0.5 threshold (for reference)
  Missed Delays→ Raw count we failed to predict (lower=better)
  Caught Delays→ Raw count we correctly predicted (higher=better)
─────────────────────────────────────────────────────────────
""")

winner = results_df.iloc[0]["Model"]
winner_file = [f for d, f, _ in model_registry if d == winner]
winner_file = winner_file[0] if winner_file else "unknown"

print(f"  ✓ SELECTED FOR TUNING: {winner}")
print(f"    → Pass MODEL_TO_TUNE = \"{winner_file}\" into hyperparameter_tuning.py")
print()