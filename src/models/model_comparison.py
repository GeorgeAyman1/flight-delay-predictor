import pandas as pd
import pickle
import numpy as np
import json
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
)


class ModelComparator:
    """
    Loads all saved models and evaluates them on the validation set.
    """

    TARGET = "departure_delayed"

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "processed"
        self.models_dir = base_dir / "models"

    def run(self) -> dict:
        """
        Evaluate all models and return comparison results.
        """
        print("Loading validation data...")
        valid = pd.read_parquet(self.data_dir / "valid_selected.parquet")
        X_valid = valid.drop(columns=[self.TARGET])
        y_valid = valid[self.TARGET]

        with open(self.models_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        X_valid_scaled = scaler.transform(X_valid)

        thresholds_path = self.models_dir / "thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                threshold_map = json.load(f)
        else:
            threshold_map = {}

        model_registry = [
            ("Logistic Regression (baseline)", "logistic_regression", True),
            ("Random Forest", "random_forest", False),
            ("Extra Trees", "extra_trees", False),
            ("XGBoost", "xgboost", False),
            ("LightGBM", "lightgbm", False),
            ("Hist Gradient Boosting", "hist_gradient_boosting", False),
            ("Gradient Boosting", "gradient_boosting", False),
            ("CatBoost", "catboost", False),
        ]

        results = []

        for display_name, filename, use_scaling in model_registry:
            model_path = self.models_dir / f"{filename}.pkl"
            if not model_path.exists():
                continue

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            Xva = X_valid_scaled if use_scaling else X_valid.values
            y_proba = model.predict_proba(Xva)[:, 1]

            thresh = threshold_map.get(filename, 0.5)
            y_pred_opt = (y_proba >= thresh).astype(int)

            roc_auc = roc_auc_score(y_valid, y_proba)
            pr_auc = average_precision_score(y_valid, y_proba)
            f1_opt = f1_score(y_valid, y_pred_opt)
            rec_opt = recall_score(y_valid, y_pred_opt)
            pre_opt = precision_score(y_valid, y_pred_opt)

            results.append({
                "Model": display_name,
                "ROC-AUC": round(roc_auc, 4),
                "PR-AUC": round(pr_auc, 4),
                "Threshold": round(thresh, 3),
                "Recall (opt)": round(rec_opt, 4),
                "Precision (opt)": round(pre_opt, 4),
                "F1 (opt)": round(f1_opt, 4),
            })

        results_df = pd.DataFrame(results).sort_values("F1 (opt)", ascending=False)
        winner = results_df.iloc[0]["Model"]
        winner_file = [f for d, f, _ in model_registry if d == winner][0]

        return {
            "results": results_df.to_dict(orient="records"),
            "winner": winner_file,
        }


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    comparator = ModelComparator(base)
    comparator.run()
