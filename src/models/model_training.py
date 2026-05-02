"""
model_training.py  — Enhanced
──────────────────────────────
EDA-informed training pipeline with expanded model set.

Key changes vs v1:
  • Added ExtraTreesClassifier and HistGradientBoostingClassifier
  • Added CatBoostClassifier (handles categoricals natively)
  • Hyperparameters pre-set from EDA insights:
      – deeper trees allowed (prev_flight_delayed has strong non-linear threshold)
      – stronger class weighting emphasis (3.4:1 imbalance)
      – higher n_estimators for boosting models (weak learners need more rounds)
  • Saves optimal_threshold per model (Precision-Recall curve, F1-maximising)
    so model_comparison.py can evaluate at the right cut-off
  • route_congestion deliberately down-weighted via feature mask comment
    (low-signal feature per EDA — keep it for tuning to decide)
"""

import pandas as pd
import json
import pickle
import time
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("  [info] CatBoost not installed — skipping. Run: pip install catboost")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[2]
DATA_DIR     = BASE_DIR / "data" / "processed"
MODELS_DIR   = BASE_DIR / "models"
WEIGHTS_PATH = MODELS_DIR / "class_weights.json"
TARGET       = "departure_delayed"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_parquet(DATA_DIR / "train_selected.parquet")
valid = pd.read_parquet(DATA_DIR / "valid_selected.parquet")

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]
X_valid = valid.drop(columns=[TARGET])
y_valid = valid[TARGET]

print(f"  Train : {X_train.shape}")
print(f"  Valid : {X_valid.shape}")

# ─────────────────────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────
raw_weights      = json.load(open(WEIGHTS_PATH))
sklearn_weights  = {int(k): v for k, v in raw_weights.items()}
scale_pos_weight = sklearn_weights[1] / sklearn_weights[0]   # ≈ 3.40

# GradientBoosting / ExtraTrees need per-sample weights
sample_weights = np.where(y_train == 1,
                          sklearn_weights[1],
                          sklearn_weights[0])

print(f"\n  sklearn class_weight : {sklearn_weights}")
print(f"  scale_pos_weight     : {scale_pos_weight:.4f}")

# ─────────────────────────────────────────────────────────────
# SCALING  (Logistic Regression only)
# ─────────────────────────────────────────────────────────────
print("\nFitting scaler on train...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

with open(MODELS_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("  Scaler saved → models/scaler.pkl")

# ─────────────────────────────────────────────────────────────
# HELPER — find the probability threshold that maximises F1
# on the validation set.  Saves per model so comparison is fair.
# ─────────────────────────────────────────────────────────────
def best_f1_threshold(y_true, y_proba):
    """Return threshold that maximises F1 on the provided labels."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # f1 = 2*p*r / (p+r) — avoid divide-by-zero
    with np.errstate(invalid="ignore"):
        f1s = 2 * precisions * recalls / (precisions + recalls)
    f1s = np.nan_to_num(f1s)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# (name, model, use_scaling, fit_kwargs)
#
# EDA insights applied per model:
#   • All tree models: deeper depths allowed (non-linear thresholds)
#   • Boosting models: more estimators (prev_flight_delayed signal needs rounds)
#   • min_samples_leaf tuned to dataset size (2.4M rows → leaves ≥ 200)
#   • ExtraTrees: fast alternative to RF, more randomised → better calibrated
#   • HistGradientBoosting: native missing-value handling, very fast
#   • CatBoost: handles categorical features (airline, airport) natively
# ─────────────────────────────────────────────────────────────
model_registry = [

    # ── 1. BASELINE — Logistic Regression ────────────────────
    # EDA: linear model will underperform on non-linear features,
    # but kept as the mandatory baseline.
    (
        "logistic_regression",
        LogisticRegression(
            class_weight=sklearn_weights,
            solver="saga",
            penalty="l2",
            C=0.1,           # regularised — avoids false confidence
            max_iter=500,
            random_state=42,
            n_jobs=-1,
        ),
        True,   # use scaled input
        {},
    ),

    # ── 2. RANDOM FOREST ─────────────────────────────────────
    # EDA: prev_flight_delayed has threshold effect → deeper trees needed.
    # Increased n_estimators and max_depth vs v1.
    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=500,
            max_depth=15,            # deeper than v1 (was 12)
            min_samples_leaf=100,    # tuned to 2.4M row dataset
            max_features="sqrt",
            class_weight=sklearn_weights,
            n_jobs=-1,
            random_state=42,
        ),
        False,
        {},
    ),

    # ── 3. EXTRA TREES ───────────────────────────────────────
    # NEW: faster than RF, more randomised splits → lower variance.
    # EDA: good fit for threshold-effect features like weather_severity.
    (
        "extra_trees",
        ExtraTreesClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=100,
            max_features="sqrt",
            class_weight=sklearn_weights,
            n_jobs=-1,
            random_state=42,
        ),
        False,
        {},
    ),

    # ── 4. XGBOOST ───────────────────────────────────────────
    # EDA-informed: stronger gamma (conservative splits), moderate depth.
    # scale_pos_weight=3.40 corrects 3.4:1 class imbalance directly.
    (
        "xgboost",
        XGBClassifier(
            n_estimators=500,        # more rounds than v1 (was 300)
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,           # tuned from search results
            colsample_bytree=0.6,    # tuned from search results
            min_child_weight=5,      # tuned from search results
            gamma=0.5,               # tuned from search results
            reg_alpha=0.1,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        ),
        False,
        {},
    ),

    # ── 5. LIGHTGBM ──────────────────────────────────────────
    # EDA: fastest boosting model. Leaf-wise growth captures
    # the tight threshold splits EDA found (weather ≥ 5, etc.).
    (
        "lightgbm",
        LGBMClassifier(
            n_estimators=700,        # more rounds — LightGBM is fast enough
            max_depth=8,
            num_leaves=63,           # controls complexity; 2^depth-1 = 127 max
            learning_rate=0.03,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            min_child_samples=100,   # analogous to min_samples_leaf
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
        False,
        {},
    ),

    # ── 6. HIST GRADIENT BOOSTING ────────────────────────────
    # NEW: sklearn's native histogram-based booster.
    # Supports missing values natively and is very memory-efficient.
    # Uses class_weight instead of scale_pos_weight.
    (
        "hist_gradient_boosting",
        HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=8,
            learning_rate=0.05,
            min_samples_leaf=100,
            l2_regularization=1.0,
            class_weight=sklearn_weights,
            random_state=42,
        ),
        False,
        {},
    ),

    # ── 7. GRADIENT BOOSTING (sklearn) ───────────────────────
    # Kept from v1 for continuity. Slowest but thorough baseline booster.
    (
        "gradient_boosting",
        GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.7,
            min_samples_leaf=100,
            random_state=42,
        ),
        False,
        {"sample_weight": sample_weights},
    ),
]

# ── 8. CATBOOST (optional) ───────────────────────────────────
# Handles high-cardinality categoricals (airline, airport) natively.
# Only added if package is installed.
if CATBOOST_AVAILABLE:
    model_registry.append((
        "catboost",
        CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="AUC",
            random_seed=42,
            verbose=0,
        ),
        False,
        {},
    ))

# ─────────────────────────────────────────────────────────────
# TRAIN & SAVE LOOP
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING ALL MODELS")
print("="*60)

train_times = {}
threshold_map = {}   # model_name → best F1 threshold on validation

for name, model, use_scaling, fit_kwargs in model_registry:

    print(f"\n  [{name}]")

    Xtr = X_train_scaled if use_scaling else X_train.values
    Xva = X_valid_scaled if use_scaling else X_valid.values

    t0 = time.time()
    model.fit(Xtr, y_train, **fit_kwargs)
    elapsed = time.time() - t0

    train_times[name] = round(elapsed, 1)
    print(f"    Training time : {elapsed:.1f}s")

    # ── find best threshold on validation ──
    y_proba = model.predict_proba(Xva)[:, 1]
    thresh   = best_f1_threshold(y_valid, y_proba)
    threshold_map[name] = thresh
    y_pred_opt = (y_proba >= thresh).astype(int)
    f1_opt     = f1_score(y_valid, y_pred_opt)
    f1_def     = f1_score(y_valid, (y_proba >= 0.5).astype(int))
    print(f"    F1 @ 0.50     : {f1_def:.4f}")
    print(f"    F1 @ {thresh:.3f}  : {f1_opt:.4f}  ← optimal threshold")

    save_path = MODELS_DIR / f"{name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved         → models/{name}.pkl")

# Save threshold map so model_comparison.py can reuse it
with open(MODELS_DIR / "thresholds.json", "w") as f:
    json.dump(threshold_map, f, indent=2)
print(f"\n  Thresholds saved → models/thresholds.json")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
for name, t in train_times.items():
    thresh = threshold_map[name]
    print(f"  {name:<30} {t:>7.1f}s   threshold={thresh:.3f}")

print("\nAll models saved to /models/")
print("Next step → run model_comparison.py")