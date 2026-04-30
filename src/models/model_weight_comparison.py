import pandas as pd
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


class WeightComparisonAnalyzer:
    """
    Compares weighted vs unweighted logistic regression on the validation set
    to demonstrate the impact of class weighting on imbalanced data.
    """

    TARGET = "departure_delayed"

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "processed"
        self.weights_path = base_dir / "models" / "class_weights.json"

    def run(self) -> dict:
        """
        Train and evaluate weighted vs unweighted logistic regression.

        Returns
        -------
        dict with keys:
            - unweighted : {"roc_auc": float, "recall": float, "f1": float}
            - weighted   : {"roc_auc": float, "recall": float, "f1": float}
        """

        # ── LOAD DATA ────────────────────────────────────────────
        print("Loading data...")
        train = pd.read_parquet(self.data_dir / "train_selected.parquet")
        valid = pd.read_parquet(self.data_dir / "valid_selected.parquet")

        X_train = train.drop(columns=[self.TARGET])
        y_train = train[self.TARGET]
        X_valid = valid.drop(columns=[self.TARGET])
        y_valid = valid[self.TARGET]

        print(f"Train: {X_train.shape}")
        print(f"Valid: {X_valid.shape}")

        # ── SCALE FEATURES (fixes convergence warning) ───────────
        # Fit ONLY on train, transform both splits
        print("\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        # ── LOAD & FIX CLASS WEIGHTS ─────────────────────────────
        # JSON stores string keys; sklearn needs keys that match the
        # actual dtype of y (float here: 0.0 / 1.0)
        raw_weights = json.load(open(self.weights_path))

        # Detect target dtype and cast keys accordingly
        if y_train.dtype == float or str(y_train.dtype).startswith("float"):
            weights = {float(k): v for k, v in raw_weights.items()}
        else:
            weights = {int(k): v for k, v in raw_weights.items()}

        print("\nClass Weights (keys cast to match target dtype):")
        print(weights)

        # ── SHARED KWARGS ────────────────────────────────────────
        # saga solver: fastest for large datasets, supports class_weight
        SHARED = dict(
            solver="saga",
            max_iter=200,
            random_state=42,
        )

        # ── 1. UNWEIGHTED MODEL ──────────────────────────────────
        print("\n" + "=" * 60)
        print("UNWEIGHTED MODEL")
        print("=" * 60)

        model_unweighted = LogisticRegression(**SHARED)
        model_unweighted.fit(X_train_scaled, y_train)

        y_pred_unweighted = model_unweighted.predict(X_valid_scaled)
        y_proba_unweighted = model_unweighted.predict_proba(X_valid_scaled)[:, 1]

        print("\nClassification Report:")
        print(classification_report(y_valid, y_pred_unweighted))
        print("ROC-AUC:", roc_auc_score(y_valid, y_proba_unweighted))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_valid, y_pred_unweighted))

        # ── 2. WEIGHTED MODEL ────────────────────────────────────
        print("\n" + "=" * 60)
        print("WEIGHTED MODEL")
        print("=" * 60)

        model_weighted = LogisticRegression(class_weight=weights, **SHARED)
        model_weighted.fit(X_train_scaled, y_train)

        y_pred_weighted = model_weighted.predict(X_valid_scaled)
        y_proba_weighted = model_weighted.predict_proba(X_valid_scaled)[:, 1]

        print("\nClassification Report:")
        print(classification_report(y_valid, y_pred_weighted))
        print("ROC-AUC:", roc_auc_score(y_valid, y_proba_weighted))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_valid, y_pred_weighted))

        # ── COMPARISON SUMMARY ───────────────────────────────────
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)

        roc_u = roc_auc_score(y_valid, y_proba_unweighted)
        roc_w = roc_auc_score(y_valid, y_proba_weighted)
        recall_u = recall_score(y_valid, y_pred_unweighted)
        recall_w = recall_score(y_valid, y_pred_weighted)
        f1_u = f1_score(y_valid, y_pred_unweighted)
        f1_w = f1_score(y_valid, y_pred_weighted)

        print(f"\n{'Metric':<30} {'Unweighted':>12} {'Weighted':>12}")
        print("-" * 56)
        print(f"{'ROC-AUC':<30} {roc_u:>12.4f} {roc_w:>12.4f}")
        print(
            f"{'Recall  (class 1 — delayed)':<30} {recall_u:>12.4f} {recall_w:>12.4f}"
        )
        print(
            f"{'F1-score (class 1 — delayed)':<30} {f1_u:>12.4f} {f1_w:>12.4f}"
        )

        print(
            """
Expected outcome:
  Unweighted → high accuracy, misses most delays (low recall)
  Weighted   → lower accuracy, catches more delays (higher recall)
  ROC-AUC should be similar — weighting shifts the threshold, not the ranking
"""
        )

        return {
            "unweighted": {"roc_auc": roc_u, "recall": recall_u, "f1": f1_u},
            "weighted": {"roc_auc": roc_w, "recall": recall_w, "f1": f1_w},
        }


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    analyzer = WeightComparisonAnalyzer(base)
    analyzer.run()