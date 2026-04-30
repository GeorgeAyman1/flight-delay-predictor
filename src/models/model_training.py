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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ModelTrainer:
    """
    Trains all baseline models, logs metrics/artifacts to MLflow,
    and saves each trained model as a .pkl file.
    """

    TARGET = "departure_delayed"

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "processed"
        self.models_dir = base_dir / "models"
        self.weights_path = self.models_dir / "class_weights.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────

    def _load_data(self):
        """Load train / validation splits and separate features from target."""
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
        """Load class weights JSON and derive sklearn / xgb / sample formats."""
        raw_weights = json.load(open(self.weights_path))
        sklearn_weights = {int(k): v for k, v in raw_weights.items()}
        scale_pos_weight = sklearn_weights[1] / sklearn_weights[0]
        print(f"\n  sklearn class_weight : {sklearn_weights}")
        print(f"  scale_pos_weight     : {scale_pos_weight:.4f}")
        return sklearn_weights, scale_pos_weight

    def _build_model_registry(self, sklearn_weights, scale_pos_weight, sample_weights):
        """
        Return the list of (name, model, use_scaling, fit_kwargs) tuples
        defining every baseline model to train.
        """
        return [
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
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Train all baseline models, log to MLflow, and save .pkl files.

        Returns
        -------
        dict with keys:
            - train_times   : {model_name: seconds}
            - models_saved  : [list of saved file paths]
        """
        X_train, y_train, X_valid, y_valid = self._load_data()
        sklearn_weights, scale_pos_weight = self._load_weights()

        # GradientBoosting has no class_weight param → per-sample weights
        sample_weights = np.where(
            y_train == 1, sklearn_weights[1], sklearn_weights[0]
        )

        # ── SCALING  (only Logistic Regression needs it) ─────────
        print("\nFitting scaler on train (used only by Logistic Regression)...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        # Save scaler so model_comparison / tuning can reuse it
        with open(self.models_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print("  Scaler saved → models/scaler.pkl")

        # ── MODEL REGISTRY ───────────────────────────────────────
        model_registry = self._build_model_registry(
            sklearn_weights, scale_pos_weight, sample_weights
        )

        # ── TRAIN & SAVE LOOP WITH MLFLOW ────────────────────────
        print("\n" + "=" * 60)
        print("TRAINING ALL MODELS")
        print("=" * 60)

        train_times = {}
        models_saved = []

        mlflow.set_tracking_uri(uri=(self.base_dir / "mlruns").as_uri())
        client = MlflowClient()
        exp_name = "flight-delay-baselines"
        exp = client.get_experiment_by_name(exp_name)

        if exp is None:
            experiment_id = client.create_experiment(name=exp_name)
        else:
            experiment_id = exp.experiment_id

        with mlflow.start_run(
            experiment_id=experiment_id, run_name="baseline-comparison"
        ) as parent_run:

            for name, model, use_scaling, fit_kwargs in model_registry:

                print(f"\n  [{name}]")

                with mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=name,
                    parent_run_id=parent_run.info.run_id,
                    nested=True,
                ):
                    # Prepare Data
                    Xtr = X_train_scaled if use_scaling else X_train.values
                    Xva = X_valid_scaled if use_scaling else X_valid.values

                    # Log Model Params
                    model_params = model.get_params()
                    mlflow.log_params(
                        {str(k): str(v) for k, v in model_params.items()}
                    )

                    # Train
                    t0 = time.time()
                    model.fit(Xtr, y_train, **fit_kwargs)
                    elapsed = time.time() - t0

                    train_times[name] = round(elapsed, 1)
                    print(f"    Training time : {elapsed:.1f}s")
                    mlflow.log_metric("training_time_s", elapsed)

                    # Metric Calculations & Validation
                    y_pred = model.predict(Xva)
                    y_proba = model.predict_proba(Xva)[:, 1]

                    mlflow.log_metric("val_roc_auc", roc_auc_score(y_valid, y_proba))
                    mlflow.log_metric("val_recall", recall_score(y_valid, y_pred))
                    mlflow.log_metric("val_precision", precision_score(y_valid, y_pred))
                    mlflow.log_metric("val_f1", f1_score(y_valid, y_pred))

                    # Confusion Matrix Artifact
                    cm = confusion_matrix(y_valid, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues")
                    ax.set_title(f"{name} — Validation Set")
                    plt.tight_layout()

                    cm_path = self.models_dir / f"cm_{name}.png"
                    fig.savefig(cm_path)
                    mlflow.log_artifact(
                        cm_path.as_posix(), artifact_path="confusion_matrices"
                    )
                    plt.close(fig)

                    # Save the pickle Model
                    save_path = self.models_dir / f"{name}.pkl"
                    with open(save_path, "wb") as f:
                        pickle.dump(model, f)
                    models_saved.append(str(save_path))
                    print(f"    Saved         → models/{name}.pkl")

        # ── SUMMARY ──────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        for name, t in train_times.items():
            print(f"  {name:<30} {t:>7.1f}s")
        print("\nAll models saved to /models/")

        return {"train_times": train_times, "models_saved": models_saved}


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    trainer = ModelTrainer(base)
    trainer.run()
