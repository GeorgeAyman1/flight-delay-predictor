import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path


class FeatureSelector:
    """
    Applies a multi-stage feature selection pipeline to the engineered features:
    non-numeric drop, variance threshold, correlation filter, mutual information,
    and Random Forest importance — with domain overrides for force-keep/force-drop.

    Saves the selected feature files to data/processed/.
    """

    # ── Configuration constants ────────────────────────────────────
    TARGET = "departure_delayed"
    RANDOM_STATE = 42
    VARIANCE_THRESHOLD = 0.01       # drop columns where < 1% of values differ
    CORRELATION_THRESHOLD = 0.80    # drop one column from any pair sharing > 64% variance
    MI_THRESHOLD = 0.001            # drop columns that contribute essentially zero information
    IMPORTANCE_THRESHOLD = 0.01     # drop columns a Random Forest considers negligible
    SAMPLE_FRACTION = 0.20          # fraction of train used for correlation and importance scoring

    # ── Domain overrides ───────────────────────────────────────────
    # EXCLUDE_FROM_VARIANCE: lookup-table features with few unique values by construction.
    # Low variance does NOT mean low information for this feature type — MI and importance
    # scoring will evaluate them properly in later steps.
    EXCLUDE_FROM_VARIANCE = [
        'airline_delay_rate',   # 5 unique values — one per airline
        'airport_delay_rate',   # 15 unique values — one per airport
        'route_delay_rate',     # ~1093 unique values — one per route
    ]

    # FORCE_KEEP: features kept regardless of importance score.
    # Domain knowledge justifies them even when the Random Forest undervalues them
    # due to limited depth or sample size.
    FORCE_KEEP = [
        'day_of_week',        # ~5% delay rate variation across week — consistent signal
        'is_holiday',         # passenger surges on federal holidays overwhelm capacity
        'is_holiday_window',  # travel surges ±2 days around major holidays
        'route_congestion',   # flights-per-runway captures capacity pressure beyond rate features
    ]

    # FORCE_DROP: features dropped regardless of importance score.
    # Better proxies for the same information already exist in the feature set.
    FORCE_DROP = [
        'latitude',    # proxy for airport identity — airport_delay_rate captures this better
        'longitude',   # proxy for airport identity — airport_delay_rate captures this better
    ]

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "processed"

        self.train_in = self.data_dir / "train_features.parquet"
        self.valid_in = self.data_dir / "valid_features.parquet"
        self.test_in = self.data_dir / "test_features.parquet"

        self.train_out = self.data_dir / "train_selected.parquet"
        self.valid_out = self.data_dir / "valid_selected.parquet"
        self.test_out = self.data_dir / "test_selected.parquet"

    def run(self) -> dict:
        """
        Run the full feature selection pipeline and save outputs.

        Returns
        -------
        dict with keys:
            - selected_features : list[str]
            - n_dropped         : int
            - train_shape       : tuple
            - valid_shape       : tuple
            - test_shape        : tuple
        """

        # ══════════════════════════════════════════════════════════
        # STEP 1 — LOAD DATA
        # ══════════════════════════════════════════════════════════

        print("Loading feature files...")
        train = pd.read_parquet(self.train_in)
        valid = pd.read_parquet(self.valid_in)
        test = pd.read_parquet(self.test_in)

        print(f"  Train: {train.shape}")
        print(f"  Valid: {valid.shape}")
        print(f"  Test:  {test.shape}")

        # ══════════════════════════════════════════════════════════
        # STEP 2 — SEPARATE TARGET
        # departure_delayed is the label, not a feature.
        # Kept separate throughout and re-attached only at save time.
        # Using .values when re-attaching avoids index misalignment bugs.
        # ══════════════════════════════════════════════════════════

        y_train = train[self.TARGET]
        y_valid = valid[self.TARGET]
        y_test = test[self.TARGET]

        X_train = train.drop(columns=[self.TARGET])
        X_valid = valid.drop(columns=[self.TARGET])
        X_test = test.drop(columns=[self.TARGET])

        print("\nAfter separating target:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_valid: {X_valid.shape}")
        print(f"  X_test:  {X_test.shape}")

        # ══════════════════════════════════════════════════════════
        # STEP 3 — DROP NON-NUMERIC COLUMNS
        # String identifier columns (carrier_code, origin_airport, destination_airport,
        # tail_number) and datetime columns (scheduled_departure_dt, date_dt) cannot be
        # fed to sklearn selectors and have no direct predictive value as raw strings.
        # Their information has already been distilled into engineered features:
        #   carrier_code        → airline_delay_rate
        #   origin_airport      → airport_delay_rate, route_delay_rate, route_congestion
        #   destination_airport → route_delay_rate, route_congestion
        #   tail_number         → prev_flight_delayed
        #   scheduled_departure_dt / date_dt → day_of_week, tod_*, is_holiday, is_holiday_window
        # ══════════════════════════════════════════════════════════

        print(f"\n{'═'*60}")
        print("STEP 3 — Drop Non-Numeric Columns")
        print(f"{'═'*60}")

        non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

        X_train = X_train.drop(columns=non_numeric_cols)
        X_valid = X_valid.drop(columns=non_numeric_cols)
        X_test = X_test.drop(columns=non_numeric_cols)

        print(f"  Dropped (non-numeric): {non_numeric_cols}")
        print(f"  Columns remaining: {len(X_train.columns)}")

        # ── Convert pandas nullable Float64 to numpy float64 ─────
        # sklearn selectors require standard numpy dtypes.
        for df in [X_train, X_valid, X_test]:
            float64_cols = df.select_dtypes(include=['Float64']).columns
            df[float64_cols] = df[float64_cols].astype('float64')

        # ── Create sample once — reused for correlation and feature importance ──
        sample = X_train.sample(frac=self.SAMPLE_FRACTION, random_state=self.RANDOM_STATE)
        y_sample = y_train.loc[sample.index]

        # ══════════════════════════════════════════════════════════
        # STEP 4 — VARIANCE THRESHOLD
        # ══════════════════════════════════════════════════════════

        print(f"\n{'═'*60}")
        print("STEP 4 — Variance Threshold")
        print(f"{'═'*60}")

        variance_candidates = [c for c in X_train.columns
                               if c not in self.EXCLUDE_FROM_VARIANCE]

        var_selector = VarianceThreshold(threshold=self.VARIANCE_THRESHOLD)
        var_selector.fit(X_train[variance_candidates])

        kept_candidates = X_train[variance_candidates].columns[var_selector.get_support()].tolist()
        low_variance_cols = X_train[variance_candidates].columns[~var_selector.get_support()].tolist()

        protected_present = [c for c in self.EXCLUDE_FROM_VARIANCE if c in X_train.columns]
        cols_after_variance = kept_candidates + protected_present

        X_train = X_train[cols_after_variance]
        X_valid = X_valid[cols_after_variance]
        X_test = X_test[cols_after_variance]
        sample = sample[cols_after_variance]

        print(f"  Protected from variance filter: {protected_present}")
        print(f"  Dropped (low variance): {low_variance_cols if low_variance_cols else 'none'}")
        print(f"  Columns remaining: {len(cols_after_variance)}")

        # ══════════════════════════════════════════════════════════
        # STEP 5 — CORRELATION FILTER
        # ══════════════════════════════════════════════════════════

        print(f"\n{'═'*60}")
        print("STEP 5 — Correlation Filter")
        print(f"{'═'*60}")

        corr_matrix = sample.corr().abs()
        target_corr = sample.corrwith(y_sample).abs()

        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop_corr = []
        for col in upper_triangle.columns:
            correlated_with = upper_triangle.index[
                upper_triangle[col] > self.CORRELATION_THRESHOLD
            ].tolist()

            for other_col in correlated_with:
                if target_corr[col] < target_corr[other_col]:
                    to_drop_corr.append(col)
                else:
                    to_drop_corr.append(other_col)

        to_drop_corr = list(set(to_drop_corr))

        X_train = X_train.drop(columns=to_drop_corr)
        X_valid = X_valid.drop(columns=to_drop_corr)
        X_test = X_test.drop(columns=to_drop_corr)
        sample = sample.drop(columns=to_drop_corr, errors='ignore')

        print(f"  Dropped (high correlation): {to_drop_corr if to_drop_corr else 'none'}")
        print(f"  Columns remaining: {len(X_train.columns)}")

        # ══════════════════════════════════════════════════════════
        # STEP 6 — MUTUAL INFORMATION
        # ══════════════════════════════════════════════════════════

        print(f"\n{'═'*60}")
        print("STEP 6 — Mutual Information")
        print(f"{'═'*60}")

        mi_scores = mutual_info_classif(
            X_train,
            y_train,
            random_state=self.RANDOM_STATE,
        )

        mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)

        print("\n  MI Scores (all features, descending):")
        for feat, score in mi_series.items():
            print(f"    {feat:<35} {score:.4f}")

        low_mi_cols = mi_series[mi_series < self.MI_THRESHOLD].index.tolist()

        X_train = X_train.drop(columns=low_mi_cols)
        X_valid = X_valid.drop(columns=low_mi_cols)
        X_test = X_test.drop(columns=low_mi_cols)
        sample = sample.drop(columns=low_mi_cols, errors='ignore')

        print(f"\n  Dropped (low MI): {low_mi_cols if low_mi_cols else 'none'}")
        print(f"  Columns remaining: {len(X_train.columns)}")

        # ══════════════════════════════════════════════════════════
        # STEP 7 — FEATURE IMPORTANCE (Random Forest)
        # ══════════════════════════════════════════════════════════

        print(f"\n{'═'*60}")
        print("STEP 7 — Feature Importance (Random Forest on 20% sample)")
        print(f"{'═'*60}")

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=self.RANDOM_STATE,
            n_jobs=-1,
        )

        rf.fit(sample, y_sample)

        importance_series = pd.Series(
            rf.feature_importances_,
            index=X_train.columns,
        ).sort_values(ascending=False)

        print("\n  Feature Importances (descending):")
        for feat, score in importance_series.items():
            bar = "█" * int(score * 200)
            flag = "  ← below threshold" if score < self.IMPORTANCE_THRESHOLD else ""
            print(f"    {feat:<35} {score:.4f}  {bar}{flag}")

        # ══════════════════════════════════════════════════════════
        # STEP 8 — FINAL DECISION
        # ══════════════════════════════════════════════════════════

        print(f"\n{'═'*60}")
        print("STEP 8 — Final Feature Selection")
        print(f"{'═'*60}")

        low_importance_cols = importance_series[
            importance_series < self.IMPORTANCE_THRESHOLD
        ].index.tolist()

        force_kept = [c for c in self.FORCE_KEEP if c in low_importance_cols]
        low_importance_cols = [c for c in low_importance_cols if c not in self.FORCE_KEEP]
        low_importance_cols += [c for c in self.FORCE_DROP if c in X_train.columns]
        low_importance_cols = list(set(low_importance_cols))

        selected_cols = [c for c in X_train.columns if c not in low_importance_cols]

        X_train = X_train[selected_cols]
        X_valid = X_valid[selected_cols]
        X_test = X_test[selected_cols]

        print(f"  Dropped (low importance):      {[c for c in low_importance_cols if c not in self.FORCE_DROP]}")
        print(f"  Dropped (domain — proxy exists): {[c for c in low_importance_cols if c in self.FORCE_DROP]}")
        print(f"  Reinstated (domain — kept):    {force_kept if force_kept else 'none'}")
        print(f"  Final feature count: {len(selected_cols)}")

        # ══════════════════════════════════════════════════════════
        # STEP 9 — RE-ATTACH TARGET AND SAVE
        # ══════════════════════════════════════════════════════════

        print(f"\n{'═'*60}")
        print("STEP 9 — Re-attach Target and Save")
        print(f"{'═'*60}")

        train_out = X_train.copy()
        valid_out = X_valid.copy()
        test_out = X_test.copy()

        train_out[self.TARGET] = y_train.values
        valid_out[self.TARGET] = y_valid.values
        test_out[self.TARGET] = y_test.values

        train_out.to_parquet(self.train_out, index=False)
        valid_out.to_parquet(self.valid_out, index=False)
        test_out.to_parquet(self.test_out, index=False)

        print(f"  Saved: {self.train_out}  ({train_out.shape[0]:,} rows, {train_out.shape[1]} cols)")
        print(f"  Saved: {self.valid_out}  ({valid_out.shape[0]:,} rows, {valid_out.shape[1]} cols)")
        print(f"  Saved: {self.test_out}   ({test_out.shape[0]:,} rows, {test_out.shape[1]} cols)")

        # ══════════════════════════════════════════════════════════
        # STEP 10 — FULL SELECTION REPORT
        # ══════════════════════════════════════════════════════════

        all_dropped = list(set(
            non_numeric_cols + low_variance_cols +
            to_drop_corr + low_mi_cols + low_importance_cols
        ))

        cols_after_non_numeric = train.shape[1] - 1 - len(non_numeric_cols)

        print(f"\n{'═'*60}")
        print("FEATURE SELECTION REPORT")
        print(f"{'═'*60}")
        print(f"\n  Started with:             {train.shape[1] - 1} features")
        print(f"  After non-numeric drop:   {cols_after_non_numeric} features  (dropped {len(non_numeric_cols)})")
        print(f"  After variance filter:    {len(cols_after_variance)} features  (dropped {len(low_variance_cols)})")
        print(f"  After correlation filter: {len(cols_after_variance) - len(to_drop_corr)} features  (dropped {len(to_drop_corr)})")
        print(f"  After MI filter:          {len(cols_after_variance) - len(to_drop_corr) - len(low_mi_cols)} features  (dropped {len(low_mi_cols)})")
        print(f"  After importance filter:  {len(selected_cols)} features  (dropped {len(low_importance_cols)})")
        print(f"\n  Total dropped: {len(all_dropped)}")

        print("\n  Dropped — non-numeric (information already in engineered features):")
        print(f"    {non_numeric_cols}")

        print("\n  Dropped — variance (near-constant, <1% minority value):")
        print(f"    {low_variance_cols if low_variance_cols else 'none'}")

        print("\n  Protected from variance (lookup-table features — few unique values by construction):")
        print(f"    {protected_present}")

        print("\n  Dropped — correlation (redundant pair, lower target correlation dropped):")
        print(f"    {to_drop_corr if to_drop_corr else 'none'}")

        print("\n  Dropped — mutual information (essentially zero target information):")
        print(f"    {low_mi_cols if low_mi_cols else 'none'}")

        print(f"\n  Dropped — importance (Random Forest score < {self.IMPORTANCE_THRESHOLD}):")
        print(f"    {[c for c in low_importance_cols if c not in self.FORCE_DROP]}")

        print("\n  Dropped — domain reasoning (better proxy already exists):")
        print(f"    {[c for c in low_importance_cols if c in self.FORCE_DROP]}")

        print("\n  Reinstated — domain reasoning (consistent signal, RF undervalued):")
        print(f"    {force_kept if force_kept else 'none'}")

        print("\n  Final selected features (ranked by importance):")
        for i, col in enumerate(selected_cols, 1):
            score = importance_series.get(col, 0.0)
            print(f"    {i:2d}. {col:<35} importance: {score:.4f}")

        print("\n  NaN check on saved train file:")
        nan_counts = train_out.isnull().sum()
        nan_counts = nan_counts[nan_counts > 0]
        if len(nan_counts) == 0:
            print("    No NaNs found ✓")
        else:
            print(nan_counts.to_string())

        print("\n  Target present in all splits:")
        print(f"    Train: {self.TARGET in train_out.columns} ✓")
        print(f"    Valid: {self.TARGET in valid_out.columns} ✓")
        print(f"    Test:  {self.TARGET in test_out.columns}  ✓")

        print(f"\n{'═'*60}")
        print("Feature selection complete.")
        print(f"{'═'*60}")

        return {
            "selected_features": selected_cols,
            "n_dropped": len(all_dropped),
            "train_shape": train_out.shape,
            "valid_shape": valid_out.shape,
            "test_shape": test_out.shape,
        }


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    selector = FeatureSelector(base)
    selector.run()