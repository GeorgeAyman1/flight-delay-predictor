import pandas as pd
import json
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    recall_score,
    f1_score,
    precision_score,
    ConfusionMatrixDisplay,
)

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


class ModelTrainer:
    """
    Handles training of all baseline models, threshold optimisation,
    and saving model artifacts.
    """

    TARGET = "departure_delayed"

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "processed"
        self.models_dir = base_dir / "models"
        self.weights_path = self.models_dir / "class_weights.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def best_f1_threshold(self, y_true, y_proba):
        """Return threshold that maximises F1 on the provided labels."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        with np.errstate(invalid="ignore"):
            f1s = 2 * precisions * recalls / (precisions + recalls)
        f1s = np.nan_to_num(f1s)
        best_idx = np.argmax(f1s)
        return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    def run(self) -> dict:
        """
        Execute the training pipeline for all models in the registry.
        """
        print("Loading data...")
        train = pd.read_parquet(self.data_dir / "train_selected.parquet")
        valid = pd.read_parquet(self.data_dir / "valid_selected.parquet")

        X_train = train.drop(columns=[self.TARGET])
        y_train = train[self.TARGET]
        X_valid = valid.drop(columns=[self.TARGET])
        y_valid = valid[self.TARGET]

        # Load weights
        raw_weights = json.load(open(self.weights_path))
        sklearn_weights = {int(k): v for k, v in raw_weights.items()}
        scale_pos_weight = sklearn_weights[1] / sklearn_weights[0]

        sample_weights = np.where(y_train == 1, sklearn_weights[1], sklearn_weights[0])

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        with open(self.models_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        model_registry = [
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight=sklearn_weights,
                    solver="saga",
                    penalty="l2",
                    C=0.1,
                    max_iter=500,
                    random_state=42,
                    n_jobs=-1,
                ),
                True,
                {},
            ),
            (
                "random_forest",
                RandomForestClassifier(
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
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.6,
                    min_child_weight=5,
                    gamma=0.5,
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
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=700,
                    max_depth=8,
                    num_leaves=63,
                    learning_rate=0.03,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.7,
                    min_child_samples=100,
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

        train_times = {}
        threshold_map = {}
        models_saved = []

        for name, model, use_scaling, fit_kwargs in model_registry:
            print(f"  [{name}]")
            Xtr = X_train_scaled if use_scaling else X_train.values
            Xva = X_valid_scaled if use_scaling else X_valid.values

            t0 = time.time()
            model.fit(Xtr, y_train, **fit_kwargs)
            elapsed = time.time() - t0
            train_times[name] = round(elapsed, 1)

            y_proba = model.predict_proba(Xva)[:, 1]
            thresh = self.best_f1_threshold(y_valid, y_proba)
            threshold_map[name] = thresh

            save_path = self.models_dir / f"{name}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            models_saved.append(str(save_path))

        with open(self.models_dir / "thresholds.json", "w") as f:
            json.dump(threshold_map, f, indent=2)

        return {
            "train_times": train_times,
            "threshold_map": threshold_map,
            "models_saved": models_saved,
        }


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    trainer = ModelTrainer(base)
    trainer.run()
