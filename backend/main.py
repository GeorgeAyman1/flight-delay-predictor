"""
Flight Delay Predictor — FastAPI Backend
Aligned to feature selection report (Team 14, Spring 2026)

Final 17 selected features:
  prev_flight_delayed, tod_early_morning, tod_evening, route_delay_rate,
  tmpf, airport_delay_rate, cloud_ceiling, airline_delay_rate, tod_afternoon,
  num_cloud_layers, mslp, relh, weather_severity, day_of_week,
  is_holiday, is_holiday_window, route_congestion

Target column : departure_delayed
Airline filter: airline_delay_rate (carrier_code was dropped in feature selection)
Time-of-day   : binary flags tod_early_morning / tod_afternoon / tod_evening
                (no dep_hour column exists in the selected feature set)

Best model    : LightGBM (tuned) — F1 0.4446, ROC-AUC 0.6760, Recall 0.6493
                Threshold 0.460  (tuned from default 0.500)
                Replaced: XGBoost tuned (F1 0.4408, ROC-AUC 0.6754)

Dataset loaded: data/processed/valid_selected.parquet  (1,314,163 rows, 18 cols)
"""

import sys
import json
from pathlib import Path
from typing import Optional

# ── Register project root on sys.path BEFORE loading any .pkl / .joblib ──────
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.preprocess import (  # noqa: F401
    IsGustyTransformer,
    NumCloudLayersTransformer,
    CloudCeilingTransformer,
    WxCodeTransformer,
    SkyC1Encoder,
)

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
MODEL_DIR = BASE / "models"
DATA_DIR  = BASE / "data" / "processed"
FRONT_DIR = BASE / "frontend"

# ── Column name constants ─────────────────────────────────────────────────────
TARGET_COL       = "departure_delayed"
DOW_COL          = "day_of_week"
TOD_COLS = {
    "Early Morning": "tod_early_morning",
    "Afternoon":     "tod_afternoon",
    "Evening":       "tod_evening",
}
AIRLINE_RATE_COL = "airline_delay_rate"

# ── Best model metadata (updated: LightGBM tuned) ────────────────────────────
BEST_MODEL_META = {
    "name":       "LightGBM (tuned)",
    "f1":         0.4446,
    "recall":     0.6493,
    "precision":  0.338,
    "roc_auc":    0.6760,
    "pr_auc":     0.4127,
    "threshold":  0.460,
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Flight Delay Predictor API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
best_model    = None
preprocessor  = None
scaler        = None
all_models    = {}
df_valid      = None
class_weights = {}

AIRLINE_RATE_MAP = {
    "Alaska":    (0.14, 0.19),
    "Delta":     (0.16, 0.21),
    "Southwest": (0.20, 0.24),
    "United":    (0.22, 0.27),
    "American":  (0.25, 0.30),
    "JetBlue":   (0.29, 0.35),
}


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_artefacts():
    global best_model, preprocessor, scaler, all_models, df_valid, class_weights

    # best_tuned_model.pkl is now LightGBM (tuned) — saved by hyperparameter_tuning_fast.py
    best_model   = joblib.load(MODEL_DIR / "best_tuned_model.pkl")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    scaler       = joblib.load(MODEL_DIR / "scaler.pkl")

    all_models = {"LightGBM (tuned)": best_model}
for name, fname in [
    ("XGBoost",             "xgboost.pkl"),
    ("CatBoost",            "catboost.pkl"),
    ("Gradient Boosting",   "gradient_boosting.pkl"),
    ("Random Forest",       "random_forest.pkl"),
    ("Logistic Regression", "logistic_regression.pkl"),
]:
    try:
        all_models[name] = joblib.load(MODEL_DIR / fname)
    except Exception as e:
        print(f"⚠ Skipping {name}: {e}")

    with open(MODEL_DIR / "class_weights.json") as f:
        class_weights = json.load(f)

    df_valid = pd.read_parquet(DATA_DIR / "valid_selected.parquet")

    print(f"✅  Dataset loaded : {len(df_valid):,} rows x {df_valid.shape[1]} columns")
    print(f"✅  Columns        : {list(df_valid.columns)}")
    print(f"✅  Models loaded  : {list(all_models.keys())}")
    print(f"✅  Best model     : {BEST_MODEL_META['name']}  "
          f"F1={BEST_MODEL_META['f1']}  ROC-AUC={BEST_MODEL_META['roc_auc']}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _filter_by_airline(df: pd.DataFrame, airline: Optional[str]) -> pd.DataFrame:
    if not airline or airline == "All Airlines":
        return df
    if AIRLINE_RATE_COL not in df.columns:
        return df
    rate_range = AIRLINE_RATE_MAP.get(airline)
    if rate_range is None:
        return df
    lo, hi = rate_range
    filtered = df[(df[AIRLINE_RATE_COL] >= lo) & (df[AIRLINE_RATE_COL] <= hi)]
    return filtered if not filtered.empty else df


def _compute_stats(df: pd.DataFrame) -> dict:
    if TARGET_COL not in df.columns:
        return {
            "total_flights": len(df), "delayed": 0, "on_time": len(df),
            "delay_rate": 0.0, "on_time_rate": 100.0, "dow": [], "tod": [],
            "error": f"Target column '{TARGET_COL}' not found."
        }

    total        = len(df)
    delayed      = int(df[TARGET_COL].sum())
    on_time      = total - delayed
    delay_rate   = round(delayed / total * 100, 1) if total else 0.0
    on_time_rate = round(100 - delay_rate, 1)

    DAY_LABELS = {1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat", 7:"Sun"}
    dow_data = []
    if DOW_COL in df.columns:
        for day in sorted(df[DOW_COL].dropna().unique()):
            sub  = df[df[DOW_COL] == day]
            rate = round(float(sub[TARGET_COL].mean()) * 100, 1)
            dow_data.append({
                "day":   int(day),
                "label": DAY_LABELS.get(int(day), str(int(day))),
                "rate":  rate,
            })

    tod_data = []
    for label, col in TOD_COLS.items():
        if col in df.columns:
            sub  = df[df[col] == 1]
            rate = round(float(sub[TARGET_COL].mean()) * 100, 1) if len(sub) else 0.0
            tod_data.append({"label": label, "rate": rate, "count": len(sub)})

    tod_flag_cols = [c for c in TOD_COLS.values() if c in df.columns]
    if tod_flag_cols:
        other_mask = (df[tod_flag_cols] == 0).all(axis=1)
        sub_other  = df[other_mask]
        if len(sub_other):
            rate = round(float(sub_other[TARGET_COL].mean()) * 100, 1)
            tod_data.append({"label": "Morning / Other", "rate": rate, "count": len(sub_other)})

    return {
        "total_flights": total, "delayed": delayed, "on_time": on_time,
        "delay_rate": delay_rate, "on_time_rate": on_time_rate,
        "dow": dow_data, "tod": tod_data,
    }


# ════════════════════════════════════════════════════════════════════════════
#  1.  STATS ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/stats")
def get_stats(airline: Optional[str] = None):
    df = _filter_by_airline(df_valid, airline)
    return _compute_stats(df)


@app.get("/api/airlines")
def get_airlines():
    return {"airlines": list(AIRLINE_RATE_MAP.keys())}


# ════════════════════════════════════════════════════════════════════════════
#  2.  DATASET EXPLORER
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/dataset")
def get_dataset(
    page:    int           = Query(1,   ge=1),
    limit:   int           = Query(50,  ge=1, le=200),
    airline: Optional[str] = None,
    search:  Optional[str] = None,
):
    df = _filter_by_airline(df_valid, airline)

    if search:
        mask = df.apply(
            lambda col: col.astype(str).str.contains(search, case=False, na=False)
        ).any(axis=1)
        df = df[mask]

    total = len(df)
    start = (page - 1) * limit
    chunk = df.iloc[start : start + limit].round(4)

    return {
        "data":    chunk.to_dict(orient="records"),
        "total":   total,
        "columns": list(chunk.columns),
        "page":    page,
        "pages":   max(1, -(-total // limit)),
    }


@app.get("/api/dataset/columns")
def get_columns():
    IMPORTANCE = {
        "prev_flight_delayed": 0.2080, "tod_early_morning": 0.1800,
        "tod_evening": 0.1180,         "route_delay_rate":  0.1011,
        "tmpf": 0.0808,                "airport_delay_rate": 0.0561,
        "cloud_ceiling": 0.0492,       "airline_delay_rate": 0.0306,
        "tod_afternoon": 0.0257,       "num_cloud_layers":   0.0233,
        "mslp": 0.0186,                "relh":               0.0148,
        "weather_severity": 0.0141,    "day_of_week":        0.0079,
        "is_holiday": 0.0003,          "is_holiday_window":  0.0017,
        "route_congestion": 0.0011,    TARGET_COL: None,
    }
    return {
        "columns": [
            {
                "name":       col,
                "dtype":      str(df_valid[col].dtype),
                "importance": IMPORTANCE.get(col),
                "is_target":  col == TARGET_COL,
            }
            for col in df_valid.columns
        ]
    }


# ════════════════════════════════════════════════════════════════════════════
#  3.  PREDICTION ENDPOINTS
#  Best model is now LightGBM (tuned) — threshold 0.460
# ════════════════════════════════════════════════════════════════════════════

class FlightFeatures(BaseModel):
    prev_flight_delayed: int
    tod_early_morning:   int
    tod_afternoon:       int
    tod_evening:         int
    route_delay_rate:    float
    airport_delay_rate:  float
    airline_delay_rate:  float
    route_congestion:    float
    tmpf:                float
    cloud_ceiling:       float
    num_cloud_layers:    int
    mslp:                float
    relh:                float
    weather_severity:    float
    day_of_week:         int
    is_holiday:          int
    is_holiday_window:   int


FEATURE_ORDER = [
    "prev_flight_delayed", "tod_early_morning", "tod_evening", "route_delay_rate",
    "tmpf", "airport_delay_rate", "cloud_ceiling", "airline_delay_rate", "tod_afternoon",
    "num_cloud_layers", "mslp", "relh", "weather_severity", "day_of_week",
    "is_holiday", "is_holiday_window", "route_congestion",
]

# Tuned threshold for best F1 on the delayed class (was 0.500 default)
BEST_THRESHOLD = BEST_MODEL_META["threshold"]  # 0.460


@app.post("/api/predict")
def predict(features: FlightFeatures):
    """Run the tuned LightGBM model at threshold 0.460."""
    import traceback as _tb
    try:
        X_raw = pd.DataFrame([features.model_dump()])[FEATURE_ORDER]
        print(f"[DEBUG] X_raw shape: {X_raw.shape}")

        try:
            X_proc = preprocessor.transform(X_raw)
        except Exception:
            print("[DEBUG] preprocessor.transform FAILED:")
            _tb.print_exc()
            X_proc = X_raw.values

        X_scal = scaler.transform(X_proc)
        prob   = float(best_model.predict_proba(X_scal)[0][1])
        pred   = int(prob >= BEST_THRESHOLD)
        risk   = "High" if prob > 0.60 else "Medium" if prob > 0.35 else "Low"

        return {
            "delayed":           bool(pred),
            "delay_probability": round(prob, 4),
            "delay_percent":     round(prob * 100, 1),
            "risk_level":        risk,
            "risk_color":        {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}[risk],
            "confidence":        round(max(prob, 1 - prob) * 100, 1),
            "threshold_used":    BEST_THRESHOLD,
            "model":             BEST_MODEL_META["name"],
        }
    except Exception as e:
        _tb.print_exc()
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.post("/api/predict/all-models")
def predict_all_models(features: FlightFeatures):
    """Run all models and return a side-by-side comparison."""
    try:
        X_raw  = pd.DataFrame([features.model_dump()])[FEATURE_ORDER]
        X_scal = scaler.transform(X_raw)

        return {
            "predictions": {
                name: {
                    "probability": round(float(m.predict_proba(X_scal)[0][1]), 4),
                    "predicted":   int(m.predict_proba(X_scal)[0][1] >= BEST_THRESHOLD),
                }
                for name, m in all_models.items()
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Multi-model prediction failed: {str(e)}")


# ════════════════════════════════════════════════════════════════════════════
#  4.  MODEL INFO  — updated to reflect LightGBM tuned as best
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/models/info")
def model_info():
    return {
        # Full comparison table — all models evaluated
        "models": [
            # ── New tuning run (LightGBM) ──────────────────────────────────
            {"name": "LightGBM (tuned)",    "f1": 0.4446, "recall": 0.6493, "precision": 0.338, "roc_auc": 0.6760, "pr_auc": 0.4127, "threshold": 0.460, "best": True,  "source": "This run"},
            {"name": "XGBoost",             "f1": 0.4442, "recall": 0.6505, "precision": 0.337, "roc_auc": 0.6758, "pr_auc": 0.4129, "threshold": 0.460, "best": False, "source": "This run"},
            {"name": "CatBoost",            "f1": 0.4431, "recall": 0.6448, "precision": 0.338, "roc_auc": 0.6748, "pr_auc": 0.4118, "threshold": 0.465, "best": False, "source": "This run"},
            {"name": "Ensemble (LGB+XGB+CB)","f1": 0.4445,"recall": 0.6391, "precision": 0.341, "roc_auc": 0.6763, "pr_auc": 0.4134, "threshold": 0.465, "best": False, "source": "This run"},
            # ── Previous report (XGBoost tuning) ──────────────────────────
            {"name": "XGBoost (tuned, prev)","f1": 0.4408, "recall": 0.5597, "precision": 0.364, "roc_auc": 0.6754, "pr_auc": None,   "threshold": 0.500, "best": False, "source": "Prev report"},
            {"name": "LightGBM (baseline)", "f1": 0.4405, "recall": 0.5594, "precision": 0.363, "roc_auc": 0.6748, "pr_auc": None,   "threshold": 0.500, "best": False, "source": "Prev report"},
            {"name": "Gradient Boosting",   "f1": 0.4390, "recall": 0.5589, "precision": 0.362, "roc_auc": 0.6733, "pr_auc": None,   "threshold": 0.500, "best": False, "source": "Prev report"},
            {"name": "Random Forest",       "f1": 0.4382, "recall": 0.5501, "precision": 0.364, "roc_auc": 0.6714, "pr_auc": None,   "threshold": 0.500, "best": False, "source": "Prev report"},
            {"name": "Logistic Regression", "f1": 0.4307, "recall": 0.5923, "precision": 0.338, "roc_auc": 0.6579, "pr_auc": None,   "threshold": 0.500, "best": False, "source": "Prev report"},
        ],
        "best_tuned": BEST_MODEL_META,
        "tuning_stages": {
            "stage1": {"method": "RandomizedSearchCV", "iters": 15, "best_f1": 0.4417, "num_leaves": 127},
            "stage2": {"method": "GridSearchCV (focused)", "combos": 27, "best_f1": 0.4417, "num_leaves": 255},
            "stage3": {"method": "Soft-voting ensemble", "members": ["LightGBM","XGBoost","CatBoost"], "best_f1": 0.4445, "winner": "single"},
        },
        "feature_importances": [
            {"feature": f, "importance": i}
            for f, i in zip(FEATURE_ORDER,
                [0.2080,0.1800,0.1180,0.1011,0.0808,0.0561,0.0492,0.0306,0.0257,
                 0.0233,0.0186,0.0148,0.0141,0.0079,0.0003,0.0017,0.0011])
        ],
        "class_weights": class_weights,
    }


# ════════════════════════════════════════════════════════════════════════════
#  5.  SERVE FRONTEND
# ════════════════════════════════════════════════════════════════════════════

if FRONT_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONT_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"message": "API running — frontend not found at " + str(FRONT_DIR)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)