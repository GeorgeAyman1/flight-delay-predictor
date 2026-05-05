import pandas as pd
import json
import pickle
import numpy as np
import time
from pathlib import Path
from sklearn.metrics import f1_score, precision_recall_curve
from catboost import CatBoostClassifier

BASE_DIR     = Path(__file__).resolve().parents[2]
DATA_DIR     = BASE_DIR / "data" / "processed"
MODELS_DIR   = BASE_DIR / "models"
WEIGHTS_PATH = MODELS_DIR / "class_weights.json"
TARGET       = "departure_delayed"

train = pd.read_parquet(DATA_DIR / "train_selected.parquet")
valid = pd.read_parquet(DATA_DIR / "valid_selected.parquet")
X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]
X_valid = valid.drop(columns=[TARGET])
y_valid = valid[TARGET]

raw_weights      = json.load(open(WEIGHTS_PATH))
sklearn_weights  = {int(k): v for k, v in raw_weights.items()}
scale_pos_weight = sklearn_weights[1] / sklearn_weights[0]

model = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.05,
    l2_leaf_reg=3.0, scale_pos_weight=scale_pos_weight,
    eval_metric="AUC", random_seed=42, verbose=0,
)

t0 = time.time()
model.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")

y_proba = model.predict_proba(X_valid.values)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_valid, y_proba)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_thresh = float(thresholds[np.argmax(f1s[:-1])])

f1_def = f1_score(y_valid, (y_proba >= 0.50).astype(int))
f1_opt = f1_score(y_valid, (y_proba >= best_thresh).astype(int))
print(f"F1 @ 0.50       : {f1_def:.4f}")
print(f"F1 @ {best_thresh:.3f}  : {f1_opt:.4f}  ← optimal threshold")

with open(MODELS_DIR / "catboost.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved → models/catboost.pkl")

# Update thresholds.json
thresholds_path = MODELS_DIR / "thresholds.json"
thresh_map = json.load(open(thresholds_path))
thresh_map["catboost"] = best_thresh
json.dump(thresh_map, open(thresholds_path, "w"), indent=2)
print("Updated → models/thresholds.json")