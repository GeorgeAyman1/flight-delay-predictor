"""
model_comparison.py
───────────────────
Loads all saved models and evaluates them on the validation set.
Produces a ranked comparison table.

Run AFTER model_training.py
"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    recall_score,
    f1_score,
    precision_score,
    ConfusionMatrixDisplay,
)


class ModelComparator:
    """
    Loads all trained baseline models, evaluates them on the validation set,
    and produces a ranked comparison table sorted by F1 score.
    """

    TARGET = "departure_delayed"

    # (display_name, filename, use_scaling)
    MODEL_REGISTRY = [
        ("Logistic Regression (baseline)", "logistic_regression", True),
        ("Random Forest", "random_forest", False),
        ("XGBoost", "xgboost", False),
        ("LightGBM", "lightgbm", False),
        ("Gradient Boosting", "gradient_boosting", False),
    ]

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "processed"
        self.models_dir = base_dir / "models"

    def run(self) -> dict:
        """
        Evaluate all saved models on the validation set and rank them.

        Returns
        -------
        dict with keys:
            - results_df : pd.DataFrame — ranked comparison table
            - winner      : str         — filename of the top model (e.g. "xgboost")
        """

        # ── LOAD VALIDATION DATA ─────────────────────────────────
        print("Loading validation data...")
        valid = pd.read_parquet(self.data_dir / "valid_selected.parquet")

        X_valid = valid.drop(columns=[self.TARGET])
        y_valid = valid[self.TARGET]

        print(f"  Valid : {X_valid.shape}")

        # ── LOAD SCALER  (for Logistic Regression only) ──────────
        with open(self.models_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        X_valid_scaled = scaler.transform(X_valid)

        # ── EVALUATE LOOP ────────────────────────────────────────
        results = []

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

            for display_name, filename, use_scaling in self.MODEL_REGISTRY:

                print(f"\n{'='*60}")
                print(f"  {display_name}")
                print(f"{'='*60}")

                with mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=filename,
                    parent_run_id=parent_run.info.run_id,
                    nested=True,
                ):
                    model_path = self.models_dir / f"{filename}.pkl"
                    if not model_path.exists():
                        print(f"  ✗ Model file not found: {model_path}")
                        print(f"    Run model_training.py first.")
                        continue

                    with open(model_path, "rb") as f:
                        model = pickle.load(f)

                    # Log parameters from loaded model
                    model_params = model.get_params()
                    mlflow.log_params(
                        {str(k): str(v) for k, v in model_params.items()}
                    )

                    Xva = X_valid_scaled if use_scaling else X_valid.values

                    y_pred = model.predict(Xva)
                    y_proba = model.predict_proba(Xva)[:, 1]

                    roc = roc_auc_score(y_valid, y_proba)
                    rec1 = recall_score(y_valid, y_pred)
                    pre1 = precision_score(y_valid, y_pred)
                    f1_1 = f1_score(y_valid, y_pred)
                    cm = confusion_matrix(y_valid, y_pred)

                    mlflow.log_metric("val_roc_auc", roc)
                    mlflow.log_metric("val_recall", rec1)
                    mlflow.log_metric("val_precision", pre1)
                    mlflow.log_metric("val_f1", f1_1)

                    # Confusion Matrix Artifact
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues")
                    ax.set_title(f"{display_name} — Validation")
                    plt.tight_layout()

                    cm_path_img = self.models_dir / f"cm_{filename}.png"
                    fig.savefig(cm_path_img)
                    mlflow.log_artifact(
                        cm_path_img.as_posix(), artifact_path="confusion_matrices"
                    )
                    plt.close(fig)

                    tn, fp, fn, tp = cm.ravel()

                print(f"\n  Classification Report:")
                print(classification_report(y_valid, y_pred, digits=3))
                print(f"  ROC-AUC         : {roc:.4f}")
                print(f"\n  Confusion Matrix:")
                print(f"    True  Negatives (correct on-time)  : {tn:>8,}")
                print(f"    False Positives (false alarm)       : {fp:>8,}")
                print(f"    False Negatives (missed delay) ← ✗ : {fn:>8,}")
                print(f"    True  Positives (caught delay) ← ✓ : {tp:>8,}")

                results.append(
                    {
                        "Model": display_name,
                        "Filename": filename,
                        "ROC-AUC": round(roc, 4),
                        "Recall (Del)": round(rec1, 4),
                        "Precision (Del)": round(pre1, 4),
                        "F1 (Del)": round(f1_1, 4),
                        "Missed Delays": fn,
                        "Caught Delays": tp,
                    }
                )

        # ── FINAL COMPARISON TABLE ───────────────────────────────
        print(f"\n{'='*60}")
        print("  FINAL COMPARISON  (sorted by F1 — delayed class)")
        print(f"{'='*60}")

        results_df = (
            pd.DataFrame(results)
            .sort_values("F1 (Del)", ascending=False)
            .reset_index(drop=True)
        )
        results_df.index += 1  # rank from 1

        display_df = results_df.drop(columns=["Filename"])
        print(display_df.to_string())

        winner_filename = results_df.iloc[0]["Filename"]
        winner_display = results_df.iloc[0]["Model"]
        print(f"  ✓ SELECTED FOR TUNING: {winner_display}")
        print()

        return {"results_df": results_df, "winner": winner_filename}


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    comparator = ModelComparator(base)
    comparator.run()