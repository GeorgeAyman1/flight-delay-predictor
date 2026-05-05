from pathlib import Path

from .preprocess import Preprocessor
from .feature_engineering import FeatureEngineer
from .feature_selection import FeatureSelector


class FeaturePipeline:
    """
    Orchestrates the full feature processing workflow end-to-end.

    Pipeline stages (executed in order):
        1. Preprocessor      — imputation, encoding, column transforms
        2. FeatureEngineer   — time, holiday, historical, lag, weather, congestion
        3. FeatureSelector   — variance, correlation, MI, importance, domain overrides

    Parameters
    ----------
    base_dir : Path or None
        Project root directory. Defaults to two levels up from this file.
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parents[2]

    def run(self) -> dict:
        """
        Execute all pipeline steps in order.

        Returns
        -------
        dict with keys for each step's results:
            - preprocessing       : dict from Preprocessor
            - feature_engineering  : dict from FeatureEngineer
            - feature_selection    : dict from FeatureSelector
        """
        pipeline_results = {}

        # ── STEP 1 — Preprocessing ──────────────────────────────
        print("  PIPELINE STEP 1/3 — PREPROCESSING")

        preprocessor = Preprocessor(self.base_dir)
        pipeline_results["preprocessing"] = preprocessor.run()

        # ── STEP 2 — Feature Engineering ────────────────────────
        print("  PIPELINE STEP 2/3 — FEATURE ENGINEERING")

        engineer = FeatureEngineer(self.base_dir)
        pipeline_results["feature_engineering"] = engineer.run()

        # ── STEP 3 — Feature Selection ──────────────────────────
        print("  PIPELINE STEP 3/3 — FEATURE SELECTION")

        selector = FeatureSelector(self.base_dir)
        pipeline_results["feature_selection"] = selector.run()

        # ── DONE ────────────────────────────────────────────────
        print("FEATURE PIPELINE COMPLETE")

        sel_result = pipeline_results["feature_selection"]
        print(f"Final features : {len(sel_result['selected_features'])}")
        print(f"Train shape    : {sel_result['train_shape']}")
        print(f"Valid shape    : {sel_result['valid_shape']}")
        print(f"Test shape     : {sel_result['test_shape']}")

        return pipeline_results


if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.run()
