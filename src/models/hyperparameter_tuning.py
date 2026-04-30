import pandas as pd
import json
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

import mlflow

from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class HyperparameterTuner:
    """
    Runs RandomizedSearchCV on a selected model using a PredefinedSplit,
    retrains the best configuration on the training set, and saves the result.
    """

    TARGET = "departure_delayed"

    def __init__(
        self,
        base_dir: Path,
        model_to_tune: str = "xgboost",
        n_iter: int = 30,
        scoring: str = "f1",
    ):
        """
        Parameters
        ----------
        base_dir : Path
            Project root directory.
        model_to_tune : str
            Which model to tune. Options: lightgbm | xgboost | random_forest
            | gradient_boosting | logistic_regression
        n_iter : int
            Number of random combinations to try.
        scoring : str
            Optimisation metric. Options: "f1" | "roc_auc" | "recall"
        """
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "processed"
        self.models_dir = base_dir / "models"
        self.weights_path = self.models_dir / "class_weights.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model_to_tune = model_to_tune
        self.n_iter = n_iter
        self.scoring = scoring

    # ─────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────

    def _load_data(self):
        """Load train / validation splits."""
        print("Loading data...")
        train = pd.read_parquet(self.data_dir / "train_selected.parquet")
        valid = pd.read_parquet(self.data_dir / "valid_selected.parquet")

        X_train = train.drop(columns=[self.TARGET])
        y_train = train[self.TARGET]
        X_valid = valid.drop(columns=[self.TARGET])
        y_valid = valid[self.TARGET]

        print(f"  Train : {X_train.shape}")
        print(f"  Valid : {X_valid.shape}")
        return X_train, y_train, X_valid, y_valid

    def _load_weights(self):
        """Load class weights and derive sklearn / xgb formats."""
        raw_weights = json.load(open(self.weights_path))
        sklearn_weights = {int(k): v for k, v in raw_weights.items()}
        scale_pos_weight = sklearn_weights[1] / sklearn_weights[0]
        return sklearn_weights, scale_pos_weight

    def _build_param_grids(self):
        """Return the hyperparameter search spaces for each model."""
        return {
            "lightgbm": {
                "n_estimators": [200, 300, 500, 700],
                "max_depth": [4, 6, 8, 10, -1],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "num_leaves": [31, 63, 127, 255],
                "subsample": [0.6, 0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
                "min_child_samples": [20, 50, 100, 200],
                "reg_alpha": [0.0, 0.1, 0.5, 1.0],
                "reg_lambda": [0.0, 0.1, 0.5, 1.0],
            },
            "xgboost": {
                "n_estimators": [200, 300, 500, 700],
                "max_depth": [4, 5, 6, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.6, 0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
                "min_child_weight": [1, 5, 10, 20],
                "gamma": [0, 0.1, 0.3, 0.5],
                "reg_alpha": [0.0, 0.1, 0.5, 1.0],
                "reg_lambda": [1.0, 2.0, 5.0, 10.0],
            },
            "random_forest": {
                "n_estimators": [200, 300, 500],
                "max_depth": [8, 10, 12, 15, None],
                "min_samples_leaf": [20, 50, 100, 200],
                "max_features": ["sqrt", "log2", 0.5],
                "min_samples_split": [2, 5, 10],
            },
            "gradient_boosting": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.5, 0.6, 0.7, 0.8],
                "min_samples_leaf": [20, 50, 100],
            },
            "logistic_regression": {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"],
                "solver": ["saga"],
            },
        }

    def _build_base_models(self, sklearn_weights, scale_pos_weight):
        """Return base model instances with fixed (non-tuned) parameters."""
        return {
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
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run hyperparameter tuning, retrain the best model, and save results.

        Returns
        -------
        dict with keys:
            - best_params  : dict  — best hyperparameters found
            - best_score   : float — best CV score
            - model_path   : str   — path to saved best tuned model
        """
        X_train, y_train, X_valid, y_valid = self._load_data()
        sklearn_weights, scale_pos_weight = self._load_weights()

        sample_weights_train = np.where(
            y_train == 1, sklearn_weights[1], sklearn_weights[0]
        )

        # ── PREDEFINED SPLIT ─────────────────────────────────────
        # sklearn's RandomizedSearchCV needs a single X and y.
        # PredefinedSplit tells it: rows with fold=-1 are TRAIN,
        # rows with fold=0 are VALIDATION. This way we use our exact
        # 2024 validation set every time instead of random CV folds.
        print("\nBuilding PredefinedSplit (train + valid combined)...")

        X_combined = pd.concat([X_train, X_valid], axis=0).reset_index(drop=True)
        y_combined = pd.concat([y_train, y_valid], axis=0).reset_index(drop=True)

        split_index = np.concatenate(
            [
                np.full(len(X_train), -1),  # train rows
                np.full(len(X_valid), 0),  # validation rows
            ]
        )
        ps = PredefinedSplit(test_fold=split_index)

        print(f"  Combined shape : {X_combined.shape}")
        print(f"  Train rows     : {(split_index == -1).sum():,}")
        print(f"  Valid rows     : {(split_index ==  0).sum():,}")

        # sample_weight for GradientBoosting must cover combined rows
        sample_weights_combined = np.concatenate(
            [sample_weights_train, np.ones(len(X_valid))]
        )

        # ── PARAM GRIDS & BASE MODELS ────────────────────────────
        param_grids = self._build_param_grids()
        base_models = self._build_base_models(sklearn_weights, scale_pos_weight)

        fit_params = {
            "gradient_boosting": {"sample_weight": sample_weights_combined},
        }

        if self.model_to_tune not in base_models:
            raise ValueError(
                f"Unknown model: '{self.model_to_tune}'. "
                f"Choose from: {list(base_models.keys())}"
            )

        # ── RUN RANDOMIZED SEARCH ────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  HYPERPARAMETER TUNING — {self.model_to_tune.upper()}")
        print(f"{'='*60}")
        print(f"  Scoring metric : {self.scoring}")
        print(f"  Iterations     : {self.n_iter}")
        print(f"  Validation     : PredefinedSplit (2024 holdout)")

        search = RandomizedSearchCV(
            estimator=base_models[self.model_to_tune],
            param_distributions=param_grids[self.model_to_tune],
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=ps,
            refit=False,
            n_jobs=-1,
            verbose=2,
            random_state=42,
        )

        extra_fit = fit_params.get(self.model_to_tune, {})

        t0 = time.time()
        search.fit(X_combined, y_combined, **extra_fit)
        elapsed = time.time() - t0

        print(f"\n  Search complete in {elapsed:.1f}s")

        # ── RESULTS ──────────────────────────────────────────────
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

        print(f"\n  Top 10 combinations (sorted by {self.scoring}):")
        for _, row in results_df.iterrows():
            print(
                f"\n  Rank {int(row['rank_test_score'])} | "
                f"Score: {row['mean_test_score']:.4f}"
            )
            for k, v in row["params"].items():
                print(f"    {k:<25} = {v}")

        best_params = search.best_params_
        best_score = search.best_score_

        print(f"\n{'='*60}")
        print("  BEST PARAMETERS FOUND")
        print(f"{'='*60}")
        for k, v in best_params.items():
            print(f"  {k:<30} = {v}")
        print(f"\n  Best {self.scoring} score on validation : {best_score:.4f}")

        # ── RETRAIN BEST MODEL ON TRAIN SPLIT ONLY ───────────────
        print(f"\n{'='*60}")
        print("  RETRAINING BEST MODEL ON TRAIN SPLIT")
        print(f"{'='*60}")

        best_model = base_models[self.model_to_tune].__class__(
            **{**base_models[self.model_to_tune].get_params(), **best_params}
        )

        train_fit_params = {
            k: v
            for k, v in fit_params.get(self.model_to_tune, {}).items()
            if k != "sample_weight"
        }
        if self.model_to_tune == "gradient_boosting":
            train_fit_params["sample_weight"] = sample_weights_train

        t0 = time.time()
        best_model.fit(X_train, y_train, **train_fit_params)
        print(f"  Retrain time : {time.time() - t0:.1f}s")

        # ── EVALUATE TUNED MODEL ON VALIDATION ───────────────────
        print(f"\n{'='*60}")
        print("  TUNED MODEL — VALIDATION RESULTS")
        print(f"{'='*60}")

        Xva = X_valid.values
        y_pred = best_model.predict(Xva)
        y_proba = best_model.predict_proba(Xva)[:, 1]

        print("\n  Classification Report:")
        print(classification_report(y_valid, y_pred, digits=3))
        print(f"  ROC-AUC : {roc_auc_score(y_valid, y_proba):.4f}")
        print(f"  Recall  : {recall_score(y_valid, y_pred):.4f}")
        print(f"  F1      : {f1_score(y_valid, y_pred):.4f}")

        # ── MLFLOW LOGGING & SAVE BEST MODEL & PARAMS ────────────
        mlflow.set_tracking_uri(uri=(self.base_dir / "mlruns").as_uri())
        exp_name = "flight-delay-baselines"
        mlflow.set_experiment(exp_name)

        with mlflow.start_run(run_name=f"{self.model_to_tune}-best-tuned"):
            mlflow.log_params(
                {str(k): str(v) for k, v in best_params.items()}
            )
            mlflow.log_metric(f"cv_{self.scoring}", best_score)

            mlflow.log_metric("val_roc_auc", roc_auc_score(y_valid, y_proba))
            mlflow.log_metric("val_recall", recall_score(y_valid, y_pred))
            mlflow.log_metric(
                "val_precision", precision_score(y_valid, y_pred, zero_division=0)
            )
            mlflow.log_metric("val_f1", f1_score(y_valid, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y_valid, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues")
            ax.set_title(f"Tuned {self.model_to_tune} — Validation")
            plt.tight_layout()
            cm_path_img = self.models_dir / f"cm_tuned_{self.model_to_tune}.png"
            fig.savefig(cm_path_img)
            mlflow.log_artifact(
                cm_path_img.as_posix(), artifact_path="confusion_matrices"
            )
            plt.close(fig)

        model_save_path = self.models_dir / "best_tuned_model.pkl"
        params_save_path = self.models_dir / "best_tuned_params.json"

        with open(model_save_path, "wb") as f:
            pickle.dump(best_model, f)

        with open(params_save_path, "w") as f:
            json.dump(
                {
                    "model": self.model_to_tune,
                    "scoring": self.scoring,
                    "n_iter": self.n_iter,
                    "best_params": {str(k): str(v) for k, v in best_params.items()},
                    "best_score": round(best_score, 4),
                },
                f,
                indent=2,
            )

        print(f"\n  Saved tuned model  → models/best_tuned_model.pkl")
        print(f"  Saved best params  → models/best_tuned_params.json")
        print(f"\n  Tuning complete ✓")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "model_path": str(model_save_path),
        }


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    tuner = HyperparameterTuner(base)
    tuner.run()
