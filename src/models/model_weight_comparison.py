import pandas as pd
import json
from pathlib import Path
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_PATH = DATA_DIR / "train_selected.parquet"
VALID_PATH = DATA_DIR / "valid_selected.parquet"

WEIGHTS_PATH = BASE_DIR / "models" / "class_weights.json"

TARGET = "departure_delayed"

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")

train = pd.read_parquet(TRAIN_PATH)
valid = pd.read_parquet(VALID_PATH)

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_valid = valid.drop(columns=[TARGET])
y_valid = valid[TARGET]

print(f"Train: {X_train.shape}")
print(f"Valid: {X_valid.shape}")

# ─────────────────────────────────────────────────────────────
# SCALE FEATURES (fixes convergence warning)
# Fit ONLY on train, transform both splits
# ─────────────────────────────────────────────────────────────
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled  = scaler.transform(X_valid)

# ─────────────────────────────────────────────────────────────
# LOAD & FIX CLASS WEIGHTS
# JSON stores string keys; sklearn needs keys that match the
# actual dtype of y (float here: 0.0 / 1.0)
# ─────────────────────────────────────────────────────────────
raw_weights = json.load(open(WEIGHTS_PATH))

# Detect target dtype and cast keys accordingly
if y_train.dtype == float or str(y_train.dtype).startswith("float"):
    weights = {float(k): v for k, v in raw_weights.items()}
else:
    weights = {int(k): v for k, v in raw_weights.items()}

print("\nClass Weights (keys cast to match target dtype):")
print(weights)

# ─────────────────────────────────────────────────────────────
# SHARED KWARGS — avoid repeating yourself
# saga solver: fastest for large datasets, supports class_weight
# ─────────────────────────────────────────────────────────────
SHARED = dict(
    solver="saga",       # best for large n; no n_jobs warning
    max_iter=200,        # saga converges faster than lbfgs on scaled data
    random_state=42,
)

# ─────────────────────────────────────────────────────────────
# 1. UNWEIGHTED MODEL
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("UNWEIGHTED MODEL")
print("="*60)

model_unweighted = LogisticRegression(**SHARED)
model_unweighted.fit(X_train_scaled, y_train)

y_pred_unweighted  = model_unweighted.predict(X_valid_scaled)
y_proba_unweighted = model_unweighted.predict_proba(X_valid_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_valid, y_pred_unweighted))
print("ROC-AUC:", roc_auc_score(y_valid, y_proba_unweighted))
print("\nConfusion Matrix:")
print(confusion_matrix(y_valid, y_pred_unweighted))


# ─────────────────────────────────────────────────────────────
# 2. WEIGHTED MODEL
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("WEIGHTED MODEL")
print("="*60)

model_weighted = LogisticRegression(class_weight=weights, **SHARED)
model_weighted.fit(X_train_scaled, y_train)

y_pred_weighted  = model_weighted.predict(X_valid_scaled)
y_proba_weighted = model_weighted.predict_proba(X_valid_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_valid, y_pred_weighted))
print("ROC-AUC:", roc_auc_score(y_valid, y_proba_weighted))
print("\nConfusion Matrix:")
print(confusion_matrix(y_valid, y_pred_weighted))


# ─────────────────────────────────────────────────────────────
# COMPARISON SUMMARY
# ─────────────────────────────────────────────────────────────
from sklearn.metrics import f1_score, recall_score

print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)

print(f"\n{'Metric':<30} {'Unweighted':>12} {'Weighted':>12}")
print("-" * 56)

roc_u = roc_auc_score(y_valid, y_proba_unweighted)
roc_w = roc_auc_score(y_valid, y_proba_weighted)

recall_u = recall_score(y_valid, y_pred_unweighted)
recall_w = recall_score(y_valid, y_pred_weighted)

f1_u = f1_score(y_valid, y_pred_unweighted)
f1_w = f1_score(y_valid, y_pred_weighted)

print(f"{'ROC-AUC':<30} {roc_u:>12.4f} {roc_w:>12.4f}")
print(f"{'Recall  (class 1 — delayed)':<30} {recall_u:>12.4f} {recall_w:>12.4f}")
print(f"{'F1-score (class 1 — delayed)':<30} {f1_u:>12.4f} {f1_w:>12.4f}")

print("""
Expected outcome:
  Unweighted → high accuracy, misses most delays (low recall)
  Weighted   → lower accuracy, catches more delays (higher recall)
  ROC-AUC should be similar — weighting shifts the threshold, not the ranking
""")