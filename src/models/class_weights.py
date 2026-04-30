import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight


class ClassWeightCalculator:
    """
    Computes balanced class weights from the training target column
    and saves them to a JSON file for downstream model training.
    """

    TARGET = "departure_delayed"

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "processed"
        self.models_dir = base_dir / "models"
        self.train_path = self.data_dir / "train_selected.parquet"
        self.weights_out = self.models_dir / "class_weights.json"

    def run(self) -> dict:
        """
        Compute and save balanced class weights.

        Returns
        -------
        dict with keys:
            - sklearn_weights : {0: float, 1: float}
            - scale_pos_weight : float  (weight[1] / weight[0])
        """

        # ══════════════════════════════════════════════════════════
        # STEP 1 — LOAD TRAINING TARGET
        # Only the target column is needed — feature columns are irrelevant here.
        # Loading only one column is significantly faster than loading all 18.
        # ══════════════════════════════════════════════════════════

        print("Loading training target...")
        y_train = pd.read_parquet(self.train_path, columns=[self.TARGET])

        total = len(y_train)
        n_delayed = int(y_train[self.TARGET].sum())
        n_ontime = total - n_delayed

        print(f"  Total rows:    {total:,}")
        print(f"  On-time  (0):  {n_ontime:,}  ({n_ontime/total*100:.1f}%)")
        print(f"  Delayed  (1):  {n_delayed:,}  ({n_delayed/total*100:.1f}%)")

        # ══════════════════════════════════════════════════════════
        # STEP 2 — COMPUTE CLASS WEIGHTS
        # Uses sklearn's compute_class_weight with strategy='balanced':
        #   weight = total_samples / (n_classes × samples_in_class)
        # Fit on training data only — validation and test must stay at natural
        # distribution to ensure evaluation metrics reflect real-world performance.
        # ══════════════════════════════════════════════════════════

        print(f"\n{'═'*60}")
        print("STEP 2 — Compute Class Weights")
        print(f"{'═'*60}")

        classes = np.array([0, 1])
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train[self.TARGET].values,
        )

        class_weights = {
            int(classes[0]): float(round(weights[0], 6)),
            int(classes[1]): float(round(weights[1], 6)),
        }

        scale_pos_weight = class_weights[1] / class_weights[0]

        print(f"  Weight for on-time  (0): {class_weights[0]:.6f}")
        print(f"  Weight for delayed  (1): {class_weights[1]:.6f}")
        print(f"  Ratio delayed/ontime:    {scale_pos_weight:.3f}x")

        # ══════════════════════════════════════════════════════════
        # STEP 3 — SAVE WEIGHTS
        # Saved as JSON so any model training script can load with one line:
        #   weights = json.load(open("models/class_weights.json"))
        # WARNING: do not combine these weights with SMOTE or any other resampling
        # method — the imbalance correction would be applied twice, causing the model
        # to massively overfit to the minority class.
        # ══════════════════════════════════════════════════════════

        print(f"\n{'═'*60}")
        print("STEP 3 — Save Weights")
        print(f"{'═'*60}")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        with open(self.weights_out, "w") as f:
            json.dump(class_weights, f, indent=2)

        print(f"  Saved: {self.weights_out}")
        print(f"\n  WARNING: never combine with SMOTE or other resampling.")

        print(f"\n{'═'*60}")
        print("Class weights complete.")
        print(f"{'═'*60}")

        return {
            "sklearn_weights": class_weights,
            "scale_pos_weight": scale_pos_weight,
        }


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    calculator = ClassWeightCalculator(base)
    calculator.run()