import pandas as pd
import json
import pickle
import numpy as np
import time
from pathlib import Path

from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    make_scorer,
    f1_score,
    recall_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — change MODEL_TO_TUNE to whichever model won
# ─────────────────────────────────────────────────────────────
MODEL_TO_TUNE = "xgboost"   # ← change to winner from model_comparison.py
                              # options: lightgbm | xgboost | random_forest
                              #          gradient_boosting | logistic_regression

N_ITER   = 30    # number of random combinations to try
                 # increase for more thorough search (slower)
                 # 30 is a good starting point for 2.4M rows

SCORING  = "f1"  # optimise for F1 on the delayed class
                 # alternatives: "roc_auc" | "recall"
                 # use "recall" if catching every delay matters more than precision

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
scale_pos_weight = sklearn_weights[1] / sklearn_weights[0]
sample_weights_train = np.where(y_train == 1,
                                sklearn_weights[1],
                                sklearn_weights[0])

# ─────────────────────────────────────────────────────────────
# PREDEFINED SPLIT
# ────────────────────────────────────────────────────────────
# sklearn's RandomizedSearchCV needs a single X and y.
# PredefinedSplit tells it: rows with fold=-1 are TRAIN,
# rows with fold=0 are VALIDATION. This way we use our exact
# 2024 validation set every time instead of random CV folds.
# ─────────────────────────────────────────────────────────────
print("\nBuilding PredefinedSplit (train + valid combined)...")

X_combined = pd.concat([X_train, X_valid], axis=0).reset_index(drop=True)
y_combined = pd.concat([y_train, y_valid], axis=0).reset_index(drop=True)

# -1 = always train, 0 = always validation
split_index = np.concatenate([
    np.full(len(X_train), -1),   # train rows
    np.full(len(X_valid),  0),   # validation rows
])
ps = PredefinedSplit(test_fold=split_index)

print(f"  Combined shape : {X_combined.shape}")
print(f"  Train rows     : {(split_index == -1).sum():,}")
print(f"  Valid rows     : {(split_index ==  0).sum():,}")

# sample_weight for GradientBoosting must cover combined rows
# (valid rows get weight 1.0 — they aren't used in fitting)
sample_weights_combined = np.concatenate([
    sample_weights_train,
    np.ones(len(X_valid)),
])

# ─────────────────────────────────────────────────────────────
# PARAM GRIDS
# Each model has its own search space.
# Ranges are intentionally wide — RandomSearch will explore them.
# ─────────────────────────────────────────────────────────────
param_grids = {

    "lightgbm": {
        "n_estimators"    : [200, 300, 500, 700],
        "max_depth"       : [4, 6, 8, 10, -1],      # -1 = no limit
        "learning_rate"   : [0.01, 0.03, 0.05, 0.1],
        "num_leaves"      : [31, 63, 127, 255],      # controls tree complexity
        "subsample"       : [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_samples": [20, 50, 100, 200],     # min data in a leaf
        "reg_alpha"       : [0.0, 0.1, 0.5, 1.0],   # L1 regularisation
        "reg_lambda"      : [0.0, 0.1, 0.5, 1.0],   # L2 regularisation
    },

    "xgboost": {
        "n_estimators"    : [200, 300, 500, 700],
        "max_depth"       : [4, 5, 6, 8],
        "learning_rate"   : [0.01, 0.03, 0.05, 0.1],
        "subsample"       : [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 5, 10, 20],
        "gamma"           : [0, 0.1, 0.3, 0.5],     # min loss reduction to split
        "reg_alpha"       : [0.0, 0.1, 0.5, 1.0],
        "reg_lambda"      : [1.0, 2.0, 5.0, 10.0],
    },

    "random_forest": {
        "n_estimators"    : [200, 300, 500],
        "max_depth"       : [8, 10, 12, 15, None],
        "min_samples_leaf": [20, 50, 100, 200],
        "max_features"    : ["sqrt", "log2", 0.5],
        "min_samples_split": [2, 5, 10],
    },

    "gradient_boosting": {
        "n_estimators"    : [100, 200, 300],
        "max_depth"       : [3, 4, 5, 6],
        "learning_rate"   : [0.01, 0.05, 0.1],
        "subsample"       : [0.5, 0.6, 0.7, 0.8],
        "min_samples_leaf": [20, 50, 100],
    },

    "logistic_regression": {
        "C"               : [0.001, 0.01, 0.1, 1.0, 10.0],  # inverse regularisation
        "penalty"         : ["l1", "l2"],
        "solver"          : ["saga"],
    },
}

# ─────────────────────────────────────────────────────────────
# BASE MODELS  (fixed params that don't get tuned)
# ─────────────────────────────────────────────────────────────
base_models = {

    "lightgbm": LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    ),

    "xgboost": XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    ),

    "random_forest": RandomForestClassifier(
        class_weight=sklearn_weights,
        n_jobs=-1,
        random_state=42,
    ),

    "gradient_boosting": GradientBoostingClassifier(
        random_state=42,
    ),

    "logistic_regression": LogisticRegression(
        class_weight=sklearn_weights,
        max_iter=200,
        random_state=42,
    ),
}

# ─────────────────────────────────────────────────────────────
# FIT PARAMS  (passed to .fit() inside RandomizedSearch)
# Only GradientBoosting needs sample_weight
# ─────────────────────────────────────────────────────────────
fit_params = {
    "gradient_boosting": {
        "sample_weight": sample_weights_combined
    },
}

# ─────────────────────────────────────────────────────────────
# RUN RANDOMIZED SEARCH
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  HYPERPARAMETER TUNING — {MODEL_TO_TUNE.upper()}")
print(f"{'='*60}")
print(f"  Scoring metric : {SCORING}")
print(f"  Iterations     : {N_ITER}")
print(f"  Validation     : PredefinedSplit (2024 holdout)")

if MODEL_TO_TUNE not in base_models:
    raise ValueError(
        f"Unknown model: '{MODEL_TO_TUNE}'. "
        f"Choose from: {list(base_models.keys())}"
    )

search = RandomizedSearchCV(
    estimator   = base_models[MODEL_TO_TUNE],
    param_distributions = param_grids[MODEL_TO_TUNE],
    n_iter      = N_ITER,
    scoring     = SCORING,
    cv          = ps,                  # our predefined train/valid split
    refit       = False,               # we'll refit manually on train only
    n_jobs      = -1,
    verbose     = 2,
    random_state= 42,
)

extra_fit = fit_params.get(MODEL_TO_TUNE, {})

t0 = time.time()
search.fit(X_combined, y_combined, **extra_fit)
elapsed = time.time() - t0

print(f"\n  Search complete in {elapsed:.1f}s")

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  SEARCH RESULTS")
print(f"{'='*60}")

results_df = pd.DataFrame(search.cv_results_)
results_df = (
    results_df[["params", "mean_test_score", "rank_test_score"]]
    .sort_values("rank_test_score")
    .head(10)
    .reset_index(drop=True)
)
results_df.index += 1

print(f"\n  Top 10 combinations (sorted by {SCORING}):")
for _, row in results_df.iterrows():
    print(f"\n  Rank {int(row['rank_test_score'])} | Score: {row['mean_test_score']:.4f}")
    for k, v in row["params"].items():
        print(f"    {k:<25} = {v}")

best_params = search.best_params_
best_score  = search.best_score_

print(f"\n{'='*60}")
print("  BEST PARAMETERS FOUND")
print(f"{'='*60}")
for k, v in best_params.items():
    print(f"  {k:<30} = {v}")
print(f"\n  Best {SCORING} score on validation : {best_score:.4f}")

# ─────────────────────────────────────────────────────────────
# RETRAIN BEST MODEL ON TRAIN SPLIT ONLY
# We never touch the validation set during training.
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  RETRAINING BEST MODEL ON TRAIN SPLIT")
print(f"{'='*60}")

# Rebuild the model with best params + fixed params merged
best_model = base_models[MODEL_TO_TUNE].__class__(
    **{**base_models[MODEL_TO_TUNE].get_params(), **best_params}
)

train_fit_params = {
    k: v for k, v in fit_params.get(MODEL_TO_TUNE, {}).items()
    if k != "sample_weight"  # we'll add train-only weights below
}
if MODEL_TO_TUNE == "gradient_boosting":
    train_fit_params["sample_weight"] = sample_weights_train

t0 = time.time()
best_model.fit(X_train, y_train, **train_fit_params)
print(f"  Retrain time : {time.time() - t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# EVALUATE TUNED MODEL ON VALIDATION
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  TUNED MODEL — VALIDATION RESULTS")
print(f"{'='*60}")

Xva = X_valid.values
y_pred  = best_model.predict(Xva)
y_proba = best_model.predict_proba(Xva)[:, 1]

print("\n  Classification Report:")
print(classification_report(y_valid, y_pred, digits=3))
print(f"  ROC-AUC : {roc_auc_score(y_valid, y_proba):.4f}")
print(f"  Recall  : {recall_score(y_valid, y_pred):.4f}")
print(f"  F1      : {f1_score(y_valid, y_pred):.4f}")

# ─────────────────────────────────────────────────────────────
# SAVE BEST MODEL & PARAMS
# ─────────────────────────────────────────────────────────────
model_save_path  = MODELS_DIR / "best_tuned_model.pkl"
params_save_path = MODELS_DIR / "best_tuned_params.json"

with open(model_save_path, "wb") as f:
    pickle.dump(best_model, f)

with open(params_save_path, "w") as f:
    json.dump({
        "model"      : MODEL_TO_TUNE,
        "scoring"    : SCORING,
        "n_iter"     : N_ITER,
        "best_params": {str(k): str(v) for k, v in best_params.items()},
        "best_score" : round(best_score, 4),
    }, f, indent=2)

print(f"\n  Saved tuned model  → models/best_tuned_model.pkl")
print(f"  Saved best params  → models/best_tuned_params.json")
print(f"\n  Tuning complete ✓")
