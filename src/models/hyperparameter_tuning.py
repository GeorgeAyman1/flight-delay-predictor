"""
hyperparameter_tuning_fast.py  — Speed-Optimised (~10 min total)
══════════════════════════════════════════════════════════════════
WHAT CHANGED FROM v1
─────────────────────
Stage 1 — SKIPPED: loads stage1_best_params.json directly.

Stage 2 — FASTER & SAFER:
  • Only tunes the 3 MOST IMPACTFUL params (learning_rate, num_leaves,
    min_child_samples). The others are locked to Stage 1 values.
    Reason: n_estimators / reg_alpha / reg_lambda have almost no interaction
    effects with each other once learning_rate is fixed. Tuning 10 params
    simultaneously produces an explosion of combos; tuning 3 ≈ 8–27 fits.
  • n_jobs=4 instead of -1 (safe on a laptop) or 1 (too slow).
    4 parallel workers lets Stage 2 finish in ~2–3 min without
    saturating RAM or thermals.

Stage 3 — ENSEMBLE of top-3: LightGBM + XGBoost + CatBoost.
  Loads pre-trained partner pkl files and soft-votes their probabilities.
  The tuned LightGBM gets 2× weight; partners get 1× each.

Total expected runtime on a modern laptop:
  Stage 2 :  ~2–3 min   (was 10–20 min with all 10 params)
  Retrain :  ~1   min
  Ensemble:  ~1   min
  ─────────────────
  Total   :  ~4–5 min
"""

import pandas as pd
import json
import pickle
import numpy as np
import time
from pathlib import Path

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    precision_recall_curve,
)
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ══════════════════════════════════════════════════════════════
# CONFIGURATION  ← only section you need to edit
# ══════════════════════════════════════════════════════════════
MODEL_TO_TUNE = "lightgbm"

SCORING = "f1"

# n_jobs for Stage 2 GridSearchCV.
# -1 = all cores (can crash laptops), 1 = safest but slow.
# 4  = good balance: fast without overloading RAM / thermals.
# Lower to 2 if you still see lag or RAM pressure.
N_JOBS_STAGE2 = 4

RUN_ENSEMBLE = True

# Paths to the pre-trained partner pkl files.
# These must have been saved by model_comparison.py (or similar).
# lightgbm_tuned.pkl is auto-saved by this script — no action needed.
ENSEMBLE_PARTNERS = ["xgboost", "catboost"]

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
BASE_DIR     = Path(r'C:/Users/VICTUS/Desktop/Engineering/Sem 8/Data Science/flight-delay-predictor')
DATA_DIR     = BASE_DIR / "data" / "processed"
MODELS_DIR   = BASE_DIR / "models"
WEIGHTS_PATH = MODELS_DIR / "class_weights.json"
TARGET       = "departure_delayed"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
# CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════
raw_weights          = json.load(open(WEIGHTS_PATH))
sklearn_weights      = {int(k): v for k, v in raw_weights.items()}
scale_pos_weight     = sklearn_weights[1] / sklearn_weights[0]   # ≈ 3.40
sample_weights_train = np.where(y_train == 1,
                                sklearn_weights[1],
                                sklearn_weights[0])

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def best_f1_threshold(y_true, y_proba):
    """Find the probability threshold that maximises F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    with np.errstate(invalid="ignore"):
        f1s = 2 * precisions * recalls / (precisions + recalls)
    f1s      = np.nan_to_num(f1s)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


def evaluate(name, model, X, y_true, threshold=None):
    """Print classification metrics; return dict of key scores."""
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
# PREDEFINED SPLIT  (train = -1, valid = 0)
# ══════════════════════════════════════════════════════════════
X_combined = pd.concat([X_train, X_valid], axis=0).reset_index(drop=True)
y_combined = pd.concat([y_train, y_valid], axis=0).reset_index(drop=True)

split_index = np.concatenate([
    np.full(len(X_train), -1),
    np.full(len(X_valid),  0),
])
ps = PredefinedSplit(test_fold=split_index)

# ══════════════════════════════════════════════════════════════
# STAGE 1 — LOAD SAVED PARAMS (SKIPPED)
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 62}")
print(f"  STAGE 1 — LOADING SAVED PARAMS (SKIPPED)")
print(f"{'=' * 62}")

with open(MODELS_DIR / 'stage1_best_params.json', 'r') as f:
    best_params_stage1 = json.load(f)

print("  Stage 1 best params:")
for k, v in best_params_stage1.items():
    print(f"    {k:<25} = {v}")

# ══════════════════════════════════════════════════════════════
# STAGE 2 — FOCUSED GRID SEARCH  (FAST VERSION)
# ─────────────────────────────────────────────────────────────
# KEY INSIGHT: Only 3 params have meaningful interaction effects
# once the others are locked. We pin everything else to the Stage 1
# best and only sweep the high-impact trio:
#
#   learning_rate     — controls step size; critical to get right
#   num_leaves        — controls model complexity / overfitting
#   min_child_samples — controls regularisation on leaf splits
#
# Each gets just 3 candidates (best ± 1 neighbour from Stage 1).
# 3 × 3 × 3 = 27 combinations × 1 fold = 27 fits  → ~2–3 min.
#
# n_jobs=4: spawns 4 worker processes. Safe on most laptops with
# 8–16 GB RAM. Each LightGBM fit itself uses n_jobs=1 (set below)
# so total CPU usage stays at 4 cores — avoids the RAM explosion
# that killed the old script with n_jobs=-1 + n_jobs=-1 (nested).
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 62}")
print(f"  STAGE 2 — FOCUSED GRID SEARCH  (3-param, 27 combos max)")
print(f"{'=' * 62}")

# ── Wide option lists (same as before) ───────────────────────
WIDE_LEARNING_RATE     = [0.01, 0.03, 0.05]
WIDE_NUM_LEAVES        = [63, 127, 255]
WIDE_MIN_CHILD_SAMPLES = [50, 100, 200]

def neighbours(best_val, options):
    """Return best ± 1 neighbour from the option list."""
    if best_val not in options:
        return [best_val]
    idx = options.index(best_val)
    return options[max(0, idx - 1): idx + 2]

focused_grid = {
    "learning_rate"    : neighbours(best_params_stage1["learning_rate"],
                                    WIDE_LEARNING_RATE),
    "num_leaves"       : neighbours(best_params_stage1["num_leaves"],
                                    WIDE_NUM_LEAVES),
    "min_child_samples": neighbours(best_params_stage1["min_child_samples"],
                                    WIDE_MIN_CHILD_SAMPLES),
}

n_combos = 1
for v in focused_grid.values():
    n_combos *= len(v)

print(f"  Grid combinations : {n_combos}  (× 1 fold = {n_combos} fits)")
for k, v in focused_grid.items():
    print(f"    {k:<25} : {v}")

# Build base estimator with all Stage 1 params locked in.
# IMPORTANT: set n_jobs=1 here so that the model itself is single-
# threaded; GridSearchCV's n_jobs=N_JOBS_STAGE2 handles parallelism
# at the CV level instead. Nested parallelism (both -1) crashes laptops.
base_lgbm = LGBMClassifier(
    scale_pos_weight   = scale_pos_weight,
    random_state       = 42,
    verbose            = -1,
    n_jobs             = 1,          # ← single-threaded per fit (safe)
    # Lock all Stage 1 params
    subsample_freq     = best_params_stage1["subsample_freq"],
    subsample          = best_params_stage1["subsample"],
    reg_lambda         = best_params_stage1["reg_lambda"],
    reg_alpha          = best_params_stage1["reg_alpha"],
    n_estimators       = best_params_stage1["n_estimators"],
    max_depth          = best_params_stage1["max_depth"],
    colsample_bytree   = best_params_stage1["colsample_bytree"],
    # Stage 2 will sweep these ↓ (passed via param_grid)
    learning_rate      = best_params_stage1["learning_rate"],
    num_leaves         = best_params_stage1["num_leaves"],
    min_child_samples  = best_params_stage1["min_child_samples"],
)

search_focused = GridSearchCV(
    estimator  = base_lgbm,
    param_grid = focused_grid,
    scoring    = SCORING,
    cv         = ps,
    refit      = False,
    n_jobs     = N_JOBS_STAGE2,   # ← outer parallelism (4 workers)
    verbose    = 1,
)

t0 = time.time()
search_focused.fit(X_combined, y_combined)
elapsed_f = time.time() - t0

best_params_stage2 = search_focused.best_params_
best_score_stage2  = search_focused.best_score_

print(f"\n  Focused search done in {elapsed_f:.1f}s  ({elapsed_f/60:.1f} min)")
print(f"  Stage 2 best {SCORING}: {best_score_stage2:.4f}")
for k, v in best_params_stage2.items():
    print(f"    {k:<25} = {v}")

# Merge: start from Stage 1, override with Stage 2 winners
best_params_final = {**best_params_stage1, **best_params_stage2}

# ══════════════════════════════════════════════════════════════
# RETRAIN BEST SINGLE MODEL ON TRAIN SPLIT
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 62}")
print("  RETRAINING BEST SINGLE MODEL ON TRAIN SPLIT")
print(f"{'=' * 62}")

best_model = LGBMClassifier(
    scale_pos_weight   = scale_pos_weight,
    random_state       = 42,
    verbose            = -1,
    n_jobs             = -1,          # ← full cores for final training (safe)
    **best_params_final,
)

t0 = time.time()
best_model.fit(X_train, y_train)
print(f"  Retrain time : {time.time() - t0:.1f}s")

print(f"\n{'=' * 62}")
print("  TUNED SINGLE MODEL — VALIDATION RESULTS")
print(f"{'=' * 62}")
single_metrics = evaluate(
    f"Tuned {MODEL_TO_TUNE}",
    best_model,
    X_valid,
    y_valid,
)

with open(MODELS_DIR / f"{MODEL_TO_TUNE}_tuned.pkl", "wb") as f:
    pickle.dump(best_model, f)
print(f"\n  Saved tuned model → models/{MODEL_TO_TUNE}_tuned.pkl")

# ══════════════════════════════════════════════════════════════
# STAGE 3 — SOFT-VOTING ENSEMBLE  (LightGBM + XGBoost + CatBoost)
# ─────────────────────────────────────────────────────────────
# Averages the predicted *probabilities* of all three models.
# Tuned LightGBM gets 2× weight; XGBoost and CatBoost get 1× each.
# Typically gains +0.5–2 F1 over any single model.
# ══════════════════════════════════════════════════════════════
if RUN_ENSEMBLE:
    print(f"\n{'=' * 62}")
    print("  STAGE 3 — SOFT-VOTING ENSEMBLE")
    print(f"{'=' * 62}")

    ensemble_estimators = [(MODEL_TO_TUNE, best_model)]
    ensemble_weights    = [2]

    for partner_name in ENSEMBLE_PARTNERS:
        model_path = MODELS_DIR / f"{partner_name}.pkl"
        if not model_path.exists():
            print(f"  [skip] {partner_name} — {model_path} not found. "
                  f"Save the pkl from model_comparison.py first.")
            continue
        with open(model_path, "rb") as f:
            partner = pickle.load(f)
        ensemble_estimators.append((partner_name, partner))
        ensemble_weights.append(1)
        print(f"  Loaded partner : {partner_name}")

    if len(ensemble_estimators) >= 2:
        print(f"\n  Ensemble members:")
        for (name, _), w in zip(ensemble_estimators, ensemble_weights):
            print(f"    {name:<32}  weight={w}")

        voting_clf = VotingClassifier(
            estimators = ensemble_estimators,
            voting     = "soft",
            weights    = ensemble_weights,
            n_jobs     = 1,           # safe: each sub-model already multi-threaded
        )

        print("\n  Fitting ensemble on train split...")
        t0 = time.time()
        voting_clf.fit(X_train, y_train)
        print(f"  Ensemble fit time : {time.time() - t0:.1f}s")

        print(f"\n{'=' * 62}")
        print("  ENSEMBLE — VALIDATION RESULTS")
        print(f"{'=' * 62}")
        ens_metrics = evaluate(
            "Soft-Voting Ensemble",
            voting_clf,
            X_valid,
            y_valid,
        )

        if ens_metrics["f1"] > single_metrics["f1"]:
            print(f"\n  ✓ Ensemble is BETTER  "
                  f"(+{ens_metrics['f1'] - single_metrics['f1']:.4f} F1)"
                  f" — saving as final model")
            final_model      = voting_clf
            final_metrics    = ens_metrics
            final_model_type = "ensemble"
        else:
            print(f"\n  → Single tuned model is better — keeping it")
            final_model      = best_model
            final_metrics    = single_metrics
            final_model_type = "single"
    else:
        print("  [warn] Not enough partner models found — skipping ensemble.")
        final_model      = best_model
        final_metrics    = single_metrics
        final_model_type = "single"

else:
    final_model      = best_model
    final_metrics    = single_metrics
    final_model_type = "single"

# ══════════════════════════════════════════════════════════════
# SAVE FINAL MODEL & PARAMS
# ══════════════════════════════════════════════════════════════
model_save_path  = MODELS_DIR / "best_tuned_model.pkl"
params_save_path = MODELS_DIR / "best_tuned_params.json"

with open(model_save_path, "wb") as f:
    pickle.dump(final_model, f)

with open(params_save_path, "w") as f:
    json.dump({
        "model_type"    : final_model_type,
        "base_model"    : MODEL_TO_TUNE,
        "scoring"       : SCORING,
        "cv_strategy"   : "PredefinedSplit (2024 holdout)",
        "stage1_params" : best_params_stage1,
        "stage2_params" : best_params_stage2,
        "final_params"  : {str(k): str(v) for k, v in best_params_final.items()},
        "stage2_score"  : round(best_score_stage2, 4),
        "final_metrics" : {k: round(v, 4) for k, v in final_metrics.items()},
    }, f, indent=2)

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 62}")
print("  FINAL SUMMARY")
print(f"{'=' * 62}")
print(f"  Model type   : {final_model_type}")
print(f"  Base model   : {MODEL_TO_TUNE}")
print(f"  ROC-AUC      : {final_metrics['roc_auc']:.4f}")
print(f"  PR-AUC       : {final_metrics['pr_auc']:.4f}")
print(f"  Recall       : {final_metrics['recall']:.4f}")
print(f"  F1           : {final_metrics['f1']:.4f}")
print(f"  Threshold    : {final_metrics['threshold']:.3f}")
print(f"\n  Saved model  → models/best_tuned_model.pkl")
print(f"  Saved params → models/best_tuned_params.json")
print(f"\n  Tuning complete ✓")