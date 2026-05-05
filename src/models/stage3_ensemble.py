"""
stage3_ensemble.py  — Soft-Voting Ensemble (Stage 3 only)
══════════════════════════════════════════════════════════
Loads the already-tuned LightGBM + XGBoost + CatBoost pkl files
and builds a soft-voting ensemble. Run this after:
  1. hyperparameter_tuning_fast.py (already done — lightgbm_tuned.pkl saved)
  2. pip install catboost  (if not already installed)

Prerequisites in models/:
  - lightgbm_tuned.pkl   ← saved by hyperparameter_tuning_fast.py
  - xgboost.pkl          ← saved by model_comparison.py
  - catboost.pkl         ← saved by model_comparison.py
"""

import pandas as pd
import json
import pickle
import numpy as np
import time
from pathlib import Path

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    precision_recall_curve,
)

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
BASE_DIR   = Path(__file__).resolve().parents[2]
DATA_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
TARGET     = "departure_delayed"

# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
print("Loading data...")
train = pd.read_parquet(DATA_DIR / "train_selected.parquet")
valid = pd.read_parquet(DATA_DIR / "valid_selected.parquet")

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]
X_valid = valid.drop(columns=[TARGET])
y_valid = valid[TARGET]

print(f"  Train : {X_train.shape}")
print(f"  Valid : {X_valid.shape}")

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def best_f1_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    with np.errstate(invalid="ignore"):
        f1s = 2 * precisions * recalls / (precisions + recalls)
    f1s = np.nan_to_num(f1s)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


def evaluate(name, model, X, y_true, threshold=None):
    y_proba = model.predict_proba(X)[:, 1]
    if threshold is None:
        threshold = best_f1_threshold(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    roc = roc_auc_score(y_true, y_proba)
    pr  = average_precision_score(y_true, y_proba)
    rec = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)

    print(f"\n  {name}")
    print(f"  {'─' * 52}")
    print(classification_report(y_true, y_pred, digits=3))
    print(f"  ROC-AUC  : {roc:.4f}")
    print(f"  PR-AUC   : {pr:.4f}")
    print(f"  Recall   : {rec:.4f}  @ threshold {threshold:.3f}")
    print(f"  F1       : {f1:.4f}")
    return {"roc_auc": roc, "pr_auc": pr, "recall": rec,
            "precision": pre, "f1": f1, "threshold": threshold}

# ══════════════════════════════════════════════════════════════
# LOAD PRE-TRAINED MODELS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 62}")
print("  LOADING PRE-TRAINED MODELS")
print(f"{'=' * 62}")

model_files = {
    "lightgbm" : MODELS_DIR / "lightgbm_tuned.pkl",
    "xgboost"  : MODELS_DIR / "xgboost.pkl",
    "catboost" : MODELS_DIR / "catboost.pkl",
}
weights = {
    "lightgbm" : 2,   # tuned winner gets 2× weight
    "xgboost"  : 1,
    "catboost" : 1,
}

loaded_models   = {}
missing_models  = []

for name, path in model_files.items():
    if not path.exists():
        print(f"  [skip] {name:<12} — {path} not found")
        missing_models.append(name)
        continue
    with open(path, "rb") as f:
        loaded_models[name] = pickle.load(f)
    print(f"  Loaded {name:<12} ← {path.name}")

if len(loaded_models) < 2:
    raise RuntimeError(
        "Need at least 2 models for an ensemble. "
        f"Missing: {missing_models}. "
        "Make sure model_comparison.py has saved the pkl files."
    )

# Show individual model scores first so you can compare
print(f"\n{'=' * 62}")
print("  INDIVIDUAL MODEL BASELINES (validation set)")
print(f"{'=' * 62}")
individual_metrics = {}
for name, model in loaded_models.items():
    individual_metrics[name] = evaluate(name, model, X_valid, y_valid)

# ══════════════════════════════════════════════════════════════
# STAGE 3 — SOFT-VOTING ENSEMBLE
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 62}")
print("  STAGE 3 — SOFT-VOTING ENSEMBLE")
print(f"{'=' * 62}")

ensemble_estimators = [(name, model) for name, model in loaded_models.items()]
ensemble_weights    = [weights[name] for name, _ in ensemble_estimators]

print("\n  Ensemble members:")
for (name, _), w in zip(ensemble_estimators, ensemble_weights):
    print(f"    {name:<32}  weight={w}")

voting_clf = VotingClassifier(
    estimators = ensemble_estimators,
    voting     = "soft",
    weights    = ensemble_weights,
    n_jobs     = 1,   # safe: each sub-model is already multi-threaded internally
)

print("\n  Fitting ensemble on train split...")
t0 = time.time()
voting_clf.fit(X_train, y_train)
print(f"  Ensemble fit time : {time.time() - t0:.1f}s")

print(f"\n{'=' * 62}")
print("  ENSEMBLE — VALIDATION RESULTS")
print(f"{'=' * 62}")
ens_metrics = evaluate("Soft-Voting Ensemble", voting_clf, X_valid, y_valid)

# ══════════════════════════════════════════════════════════════
# PICK & SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════
best_individual_f1   = max(m["f1"] for m in individual_metrics.values())
best_individual_name = max(individual_metrics, key=lambda n: individual_metrics[n]["f1"])

if ens_metrics["f1"] > best_individual_f1:
    print(f"\n  ✓ Ensemble is BETTER "
          f"(+{ens_metrics['f1'] - best_individual_f1:.4f} F1 vs {best_individual_name})"
          f" — saving as final model")
    final_model      = voting_clf
    final_metrics    = ens_metrics
    final_model_type = "ensemble"
else:
    print(f"\n  → Best individual model ({best_individual_name}) is better — keeping it")
    final_model      = loaded_models[best_individual_name]
    final_metrics    = individual_metrics[best_individual_name]
    final_model_type = "single"

with open(MODELS_DIR / "best_tuned_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open(MODELS_DIR / "best_tuned_params.json", "w") as f:
    json.dump({
        "model_type"    : final_model_type,
        "ensemble_members": list(loaded_models.keys()),
        "ensemble_weights": weights,
        "final_metrics" : {k: round(v, 4) for k, v in final_metrics.items()},
        "individual_metrics": {
            name: {k: round(v, 4) for k, v in m.items()}
            for name, m in individual_metrics.items()
        },
    }, f, indent=2)

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 62}")
print("  FINAL SUMMARY")
print(f"{'=' * 62}")
print(f"  Model type   : {final_model_type}")
print(f"  ROC-AUC      : {final_metrics['roc_auc']:.4f}")
print(f"  PR-AUC       : {final_metrics['pr_auc']:.4f}")
print(f"  Recall       : {final_metrics['recall']:.4f}")
print(f"  F1           : {final_metrics['f1']:.4f}")
print(f"  Threshold    : {final_metrics['threshold']:.3f}")
print("\n  Saved model  → models/best_tuned_model.pkl")
print("  Saved params → models/best_tuned_params.json")
print("\n  Stage 3 complete ✓")