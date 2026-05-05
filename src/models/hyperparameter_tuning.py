import pandas as pd
import json
import pickle
import numpy as np
from pathlib import Path

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    precision_recall_curve,
)
from lightgbm import LGBMClassifier


class HyperparameterTuner:
    """
    Optimises hyperparameters for a specified model and optionally
    creates an ensemble.
    """

    TARGET = "departure_delayed"

    def __init__(
        self,
        base_dir: Path,
        model_to_tune: str = "lightgbm",
        n_iter: int = 30,
        scoring: str = "f1",
    ):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "processed"
        self.models_dir = base_dir / "models"
        self.weights_path = self.models_dir / "class_weights.json"
        self.model_to_tune = model_to_tune
        self.n_iter = n_iter
        self.scoring = scoring

    def best_f1_threshold(self, y_true, y_proba):
        """Find the probability threshold that maximises F1."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        with np.errstate(invalid="ignore"):
            f1s = 2 * precisions * recalls / (precisions + recalls)
        f1s = np.nan_to_num(f1s)
        best_idx = np.argmax(f1s)
        return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    def run(self) -> dict:
        """
        Execute the tuning pipeline.
        """
        print("Loading data...")
        train = pd.read_parquet(self.data_dir / "train_selected.parquet")
        valid = pd.read_parquet(self.data_dir / "valid_selected.parquet")

        X_train = train.drop(columns=[self.TARGET])
        y_train = train[self.TARGET]
        X_valid = valid.drop(columns=[self.TARGET])
        y_valid = valid[self.TARGET]

        raw_weights = json.load(open(self.weights_path))
        sklearn_weights = {int(k): v for k, v in raw_weights.items()}
        scale_pos_weight = sklearn_weights[1] / sklearn_weights[0]

        # Fast tuning logic (Stage 2 focused)
        with open(self.models_dir / "stage1_best_params.json", "r") as f:
            best_params_stage1 = json.load(f)

        focused_grid = {
            "learning_rate": [0.01, 0.03, 0.05],
            "num_leaves": [63, 127, 255],
            "min_child_samples": [50, 100, 200],
        }

        X_combined = pd.concat([X_train, X_valid], axis=0).reset_index(drop=True)
        y_combined = pd.concat([y_train, y_valid], axis=0).reset_index(drop=True)
        split_index = np.concatenate([np.full(len(X_train), -1), np.full(len(X_valid), 0)])
        ps = PredefinedSplit(test_fold=split_index)

        base_lgbm = LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
            n_jobs=1,
            **{k: v for k, v in best_params_stage1.items() if k not in focused_grid}
        )

        search = GridSearchCV(
            estimator=base_lgbm,
            param_grid=focused_grid,
            scoring=self.scoring,
            cv=ps,
            refit=False,
            n_jobs=4,
            verbose=1,
        )

        search.fit(X_combined, y_combined)
        best_params_final = {**best_params_stage1, **search.best_params_}

        best_model = LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
            **best_params_final,
        )
        best_model.fit(X_train, y_train)

        save_path = self.models_dir / "best_tuned_model.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(best_model, f)

        return {
            "model_path": str(save_path),
            "best_params": best_params_final,
        }


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    tuner = HyperparameterTuner(base)
    tuner.run()
