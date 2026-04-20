import pandas as pd
import json
import pickle
import time
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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
raw_weights = json.load(open(WEIGHTS_PATH))

# sklearn needs keys that match y dtype (int here after cast)
sklearn_weights  = {int(k): v for k, v in raw_weights.items()}

# XGBoost / LightGBM use a single ratio instead
scale_pos_weight = sklearn_weights[1] / sklearn_weights[0]

# GradientBoosting has no class_weight param → per-sample weights
sample_weights = np.where(y_train == 1,
                          sklearn_weights[1],
                          sklearn_weights[0])

print(f"\n  sklearn class_weight : {sklearn_weights}")
print(f"  scale_pos_weight     : {scale_pos_weight:.4f}")

# ─────────────────────────────────────────────────────────────
# SCALING  (only Logistic Regression needs it)
# ─────────────────────────────────────────────────────────────
print("\nFitting scaler on train (used only by Logistic Regression)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled  = scaler.transform(X_valid)

# Save scaler so model_comparison / tuning can reuse it
with open(MODELS_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("  Scaler saved → models/scaler.pkl")

# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# (name, model, use_scaling, fit_kwargs)
# ─────────────────────────────────────────────────────────────
model_registry = [

    # ── 1. BASELINE ──────────────────────────────────────────
    (
        "logistic_regression",
        LogisticRegression(
            class_weight=sklearn_weights,
            solver="saga",
            max_iter=200,
            random_state=42,
        ),
        True,
        {},
    ),

    # ── 2. RANDOM FOREST ─────────────────────────────────────
    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=50,
            class_weight=sklearn_weights,
            n_jobs=-1,
            random_state=42,
        ),
        False,
        {},
    ),

    # ── 3. XGBOOST ───────────────────────────────────────────
    (
        "xgboost",
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
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

    # ── 4. LIGHTGBM ──────────────────────────────────────────
    (
        "lightgbm",
        LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
        False,
        {},
    ),

    # ── 5. GRADIENT BOOSTING ─────────────────────────────────
    (
        "gradient_boosting",
        GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.5,
            random_state=42,
        ),
        False,
        {"sample_weight": sample_weights},
    ),
]

# ─────────────────────────────────────────────────────────────
# TRAIN & SAVE LOOP
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING ALL MODELS")
print("="*60)

train_times = {}

for name, model, use_scaling, fit_kwargs in model_registry:

    print(f"\n  [{name}]")

    Xtr = X_train_scaled if use_scaling else X_train.values
    
    t0 = time.time()
    model.fit(Xtr, y_train, **fit_kwargs)
    elapsed = time.time() - t0

    train_times[name] = round(elapsed, 1)
    print(f"    Training time : {elapsed:.1f}s")

    save_path = MODELS_DIR / f"{name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved         → models/{name}.pkl")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
for name, t in train_times.items():
    print(f"  {name:<30} {t:>7.1f}s")

print("\nAll models saved to /models/")
