"""
model_pipeline.py
─────────────────
Orchestrates the full modeling workflow end-to-end by calling each
model stage class in sequence.

Usage:
    from src.models import ModelPipeline

    pipeline = ModelPipeline()
    results = pipeline.run()

    # Or with custom settings:
    pipeline = ModelPipeline(
        model_to_tune="lightgbm",
        n_iter=50,
        scoring="recall",
        include_weight_comparison=True,
    )
    results = pipeline.run()
"""

from pathlib import Path

from .class_weights import ClassWeightCalculator
from .model_training import ModelTrainer
from .model_comparison import ModelComparator
from .model_weight_comparison import WeightComparisonAnalyzer
from .hyperparameter_tuning import HyperparameterTuner


class ModelPipeline:
    """
    Orchestrates the full modeling workflow end-to-end.

    Pipeline stages (executed in order):
        1. ClassWeightCalculator  — compute balanced class weights
        2. ModelTrainer           — train all baseline models
        3. ModelComparator        — evaluate & rank models on validation set
        4. WeightComparisonAnalyzer (optional) — weighted vs unweighted LR
        5. HyperparameterTuner   — tune the best (or specified) model

    Parameters
    ----------
    base_dir : Path or None
        Project root directory. Defaults to two levels up from this file.
    model_to_tune : str or None
        Which model to tune in step 5. If None, automatically uses
        the winner from step 3 (ModelComparator).
    n_iter : int
        Number of random hyperparameter combinations to try (default 30).
    scoring : str
        Optimisation metric for tuning: "f1" | "roc_auc" | "recall".
    include_weight_comparison : bool
        If True, run the WeightComparisonAnalyzer as step 4 (default False).
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        model_to_tune: str | None = None,
        n_iter: int = 30,
        scoring: str = "f1",
        include_weight_comparison: bool = False,
    ):
        self.base_dir = base_dir or Path(__file__).resolve().parents[2]
        self.model_to_tune = model_to_tune
        self.n_iter = n_iter
        self.scoring = scoring
        self.include_weight_comparison = include_weight_comparison

    def run(self) -> dict:
        """
        Execute all pipeline steps in order.

        Returns
        -------
        dict with keys for each step's results:
            - class_weights       : dict from ClassWeightCalculator
            - training            : dict from ModelTrainer
            - comparison          : dict from ModelComparator
            - weight_comparison   : dict from WeightComparisonAnalyzer (if enabled)
            - tuning              : dict from HyperparameterTuner
        """
        pipeline_results = {}

        # ── STEP 1 — Class Weights ───────────────────────────────
        print("\n" + "█" * 60)
        print("  PIPELINE STEP 1/5 — CLASS WEIGHTS")
        print("█" * 60)

        weight_calc = ClassWeightCalculator(self.base_dir)
        pipeline_results["class_weights"] = weight_calc.run()

        # ── STEP 2 — Model Training ─────────────────────────────
        print("PIPELINE STEP 2/5 — MODEL TRAINING")

        trainer = ModelTrainer(self.base_dir)
        pipeline_results["training"] = trainer.run()

        # ── STEP 3 — Model Comparison ────────────────────────────
        print("PIPELINE STEP 3/5 — MODEL COMPARISON")

        comparator = ModelComparator(self.base_dir)
        comparison_results = comparator.run()
        pipeline_results["comparison"] = comparison_results

        # ── STEP 4 — Weight Comparison (optional) ────────────────
        if self.include_weight_comparison:
            print("PIPELINE STEP 4/5 — WEIGHT COMPARISON (optional)")

            analyzer = WeightComparisonAnalyzer(self.base_dir)
            pipeline_results["weight_comparison"] = analyzer.run()
        else:
            print("PIPELINE STEP 4/5 — WEIGHT COMPARISON (skipped)")

        # ── STEP 5 — Hyperparameter Tuning ───────────────────────
        print("PIPELINE STEP 5/5 — HYPERPARAMETER TUNING")

        # If no model specified, use the winner from comparison
        model_to_tune = self.model_to_tune or comparison_results["winner"]
        print(f"  Model to tune: {model_to_tune}")

        tuner = HyperparameterTuner(
            base_dir=self.base_dir,
            model_to_tune=model_to_tune,
            n_iter=self.n_iter,
            scoring=self.scoring,
        )
        pipeline_results["tuning"] = tuner.run()

        # ── DONE ─────────────────────────────────────────────────
        print("PIPELINE COMPLETE")
        print(f"Winner           : {model_to_tune}")
        print(f"Tuned model saved: {pipeline_results['tuning']['model_path']}")

        return pipeline_results


if __name__ == "__main__":
    pipeline = ModelPipeline()
    pipeline.run()
