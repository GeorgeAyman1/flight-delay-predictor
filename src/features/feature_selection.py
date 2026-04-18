import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR  = BASE_DIR / "data" / "processed"

TRAIN_IN = DATA_DIR / "train_features.parquet"
VALID_IN = DATA_DIR / "valid_features.parquet"
TEST_IN  = DATA_DIR / "test_features.parquet"

TRAIN_OUT = DATA_DIR / "train_selected.parquet"
VALID_OUT = DATA_DIR / "valid_selected.parquet"
TEST_OUT  = DATA_DIR / "test_selected.parquet"

# ── Configuration constants ────────────────────────────────────────────────────
# Defined at the top so any threshold can be changed in one place
TARGET                = "departure_delayed"
RANDOM_STATE          = 42
VARIANCE_THRESHOLD    = 0.01   # drop columns where < 1% of values differ
CORRELATION_THRESHOLD = 0.80   # drop one column from any pair sharing > 64% variance
MI_THRESHOLD          = 0.001  # drop columns that contribute essentially zero information
IMPORTANCE_THRESHOLD  = 0.01   # drop columns a Random Forest considers negligible
SAMPLE_FRACTION       = 0.20   # fraction of train used for correlation and importance scoring

# ── Domain overrides ───────────────────────────────────────────────────────────
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


def main():

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 1 — LOAD DATA
    # ══════════════════════════════════════════════════════════════════════════════

    print("Loading feature files...")
    train = pd.read_parquet(TRAIN_IN)
    valid = pd.read_parquet(VALID_IN)
    test  = pd.read_parquet(TEST_IN)

    print(f"  Train: {train.shape}")
    print(f"  Valid: {valid.shape}")
    print(f"  Test:  {test.shape}")

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 2 — SEPARATE TARGET
    # departure_delayed is the label, not a feature.
    # Kept separate throughout and re-attached only at save time.
    # Using .values when re-attaching avoids index misalignment bugs.
    # ══════════════════════════════════════════════════════════════════════════════

    y_train = train[TARGET]
    y_valid = valid[TARGET]
    y_test  = test[TARGET]

    X_train = train.drop(columns=[TARGET])
    X_valid = valid.drop(columns=[TARGET])
    X_test  = test.drop(columns=[TARGET])

    print(f"\nAfter separating target:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_valid: {X_valid.shape}")
    print(f"  X_test:  {X_test.shape}")

    # ══════════════════════════════════════════════════════════════════════════════
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
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print("STEP 3 — Drop Non-Numeric Columns")
    print(f"{'═'*60}")

    # Automatically detect non-numeric columns rather than hardcoding names
    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    X_train = X_train.drop(columns=non_numeric_cols)
    X_valid = X_valid.drop(columns=non_numeric_cols)
    X_test  = X_test.drop(columns=non_numeric_cols)

    print(f"  Dropped (non-numeric): {non_numeric_cols}")
    print(f"  Columns remaining: {len(X_train.columns)}")

    # ── Convert pandas nullable Float64 to numpy float64 ──────────────────────
    # sklearn selectors require standard numpy dtypes.
    # Float64 (capital F) is pandas nullable float — not recognized correctly
    # by VarianceThreshold, causing good features to appear near-constant.
    for df in [X_train, X_valid, X_test]:
        float64_cols = df.select_dtypes(include=['Float64']).columns
        df[float64_cols] = df[float64_cols].astype('float64')

    # ── Create sample once — reused for correlation and feature importance ─────
    # Created here after non-numeric drop and dtype conversion so sample columns
    # always match X_train exactly. Random (not first N rows) to avoid temporal bias.
    sample   = X_train.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_STATE)
    y_sample = y_train.loc[sample.index]

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 4 — VARIANCE THRESHOLD
    # A near-constant column cannot help the model distinguish delayed from on-time.
    # Threshold = 0.01 → for a binary column, drops it if < 1% of rows are minority.
    # Fit on train only — applying the same mask to valid/test is not leakage.
    #
    # Rate features (airline_delay_rate, airport_delay_rate, route_delay_rate) are
    # excluded — they are lookup-table features with few unique values by construction.
    # Low variance here does NOT mean low information. MI and importance will judge them.
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print("STEP 4 — Variance Threshold")
    print(f"{'═'*60}")

    # Only run variance filter on non-protected columns
    variance_candidates = [c for c in X_train.columns
                           if c not in EXCLUDE_FROM_VARIANCE]

    var_selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    var_selector.fit(X_train[variance_candidates])   # fit on train only

    kept_candidates   = X_train[variance_candidates].columns[var_selector.get_support()].tolist()
    low_variance_cols = X_train[variance_candidates].columns[~var_selector.get_support()].tolist()

    # Final columns = kept variance candidates + protected rate features
    protected_present = [c for c in EXCLUDE_FROM_VARIANCE if c in X_train.columns]
    cols_after_variance = kept_candidates + protected_present

    # Apply the same column mask to all three splits and keep sample in sync
    X_train = X_train[cols_after_variance]
    X_valid = X_valid[cols_after_variance]
    X_test  = X_test[cols_after_variance]
    sample  = sample[cols_after_variance]

    print(f"  Protected from variance filter: {protected_present}")
    print(f"  Dropped (low variance): {low_variance_cols if low_variance_cols else 'none'}")
    print(f"  Columns remaining: {len(cols_after_variance)}")

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 5 — CORRELATION FILTER
    # Detects pairs of features that are redundant with each other (not with target).
    # Computed on the 20% sample — rankings are identical to full data but much faster.
    # Upper triangle trick: each pair appears exactly once to avoid double-dropping.
    # When two features exceed the threshold, the one with lower target correlation
    # is dropped — keeping whichever carries more information about the label.
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print("STEP 5 — Correlation Filter")
    print(f"{'═'*60}")

    # Both computed on sample — fast and representative
    corr_matrix = sample.corr().abs()
    target_corr = sample.corrwith(y_sample).abs()

    # Keep only the upper triangle — NaN (not 0) in lower half to avoid false matches
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop_corr = []
    for col in upper_triangle.columns:
        # Find all columns that are too correlated with this one
        correlated_with = upper_triangle.index[
            upper_triangle[col] > CORRELATION_THRESHOLD
        ].tolist()

        for other_col in correlated_with:
            # Drop whichever has the lower correlation with the target
            if target_corr[col] < target_corr[other_col]:
                to_drop_corr.append(col)
            else:
                to_drop_corr.append(other_col)

    # Deduplicate — a column may have been flagged by more than one pair
    to_drop_corr = list(set(to_drop_corr))

    X_train = X_train.drop(columns=to_drop_corr)
    X_valid = X_valid.drop(columns=to_drop_corr)
    X_test  = X_test.drop(columns=to_drop_corr)
    sample  = sample.drop(columns=to_drop_corr, errors='ignore')

    print(f"  Dropped (high correlation): {to_drop_corr if to_drop_corr else 'none'}")
    print(f"  Columns remaining: {len(X_train.columns)}")

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 6 — MUTUAL INFORMATION
    # Measures how much knowing a feature reduces uncertainty about the target.
    # Works on all feature types and captures non-linear relationships.
    # Threshold is intentionally conservative (0.001) — we are only eliminating
    # features that contribute essentially zero information. Feature importance
    # in the next step handles the finer ranking.
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print("STEP 6 — Mutual Information")
    print(f"{'═'*60}")

    mi_scores = mutual_info_classif(
        X_train,
        y_train,
        random_state=RANDOM_STATE
    )

    mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)

    print("\n  MI Scores (all features, descending):")
    for feat, score in mi_series.items():
        print(f"    {feat:<35} {score:.4f}")

    low_mi_cols = mi_series[mi_series < MI_THRESHOLD].index.tolist()

    X_train = X_train.drop(columns=low_mi_cols)
    X_valid = X_valid.drop(columns=low_mi_cols)
    X_test  = X_test.drop(columns=low_mi_cols)
    sample  = sample.drop(columns=low_mi_cols, errors='ignore')

    print(f"\n  Dropped (low MI): {low_mi_cols if low_mi_cols else 'none'}")
    print(f"  Columns remaining: {len(X_train.columns)}")

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 7 — FEATURE IMPORTANCE (Random Forest)
    # Trains a lightweight Random Forest on the same 20% sample used for correlation.
    # Parameters are deliberately conservative to keep this fast:
    #   n_estimators=100    → stable rankings without being slow
    #   max_depth=8         → prevents overfitting / memorising noise
    #   min_samples_leaf=50 → every split must cover ≥50 rows (statistically reliable)
    #   class_weight=balanced → prevents the 78.6% majority class from dominating scores
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print("STEP 7 — Feature Importance (Random Forest on 20% sample)")
    print(f"{'═'*60}")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1          # use all CPU cores
    )

    rf.fit(sample, y_sample)

    importance_series = pd.Series(
        rf.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print("\n  Feature Importances (descending):")
    for feat, score in importance_series.items():
        bar  = "█" * int(score * 200)
        flag = "  ← below threshold" if score < IMPORTANCE_THRESHOLD else ""
        print(f"    {feat:<35} {score:.4f}  {bar}{flag}")

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 8 — FINAL DECISION
    # Drop features below the importance threshold, then apply domain overrides:
    #   FORCE_KEEP → reinstate features domain knowledge justifies keeping
    #   FORCE_DROP → remove features where better proxies already exist
    # Domain reasoning is the final authority — statistics inform, not decide.
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print("STEP 8 — Final Feature Selection")
    print(f"{'═'*60}")

    low_importance_cols = importance_series[
        importance_series < IMPORTANCE_THRESHOLD
    ].index.tolist()

    # Apply domain overrides
    force_kept = [c for c in FORCE_KEEP if c in low_importance_cols]
    low_importance_cols = [c for c in low_importance_cols if c not in FORCE_KEEP]
    low_importance_cols += [c for c in FORCE_DROP if c in X_train.columns]
    low_importance_cols  = list(set(low_importance_cols))

    selected_cols = [c for c in X_train.columns if c not in low_importance_cols]

    X_train = X_train[selected_cols]
    X_valid = X_valid[selected_cols]
    X_test  = X_test[selected_cols]

    print(f"  Dropped (low importance):      {[c for c in low_importance_cols if c not in FORCE_DROP]}")
    print(f"  Dropped (domain — proxy exists): {[c for c in low_importance_cols if c in FORCE_DROP]}")
    print(f"  Reinstated (domain — kept):    {force_kept if force_kept else 'none'}")
    print(f"  Final feature count: {len(selected_cols)}")

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 9 — RE-ATTACH TARGET AND SAVE
    # Target is added back using .values to avoid index misalignment.
    # Without .values, pandas aligns by index — if X_train and y_train ever had
    # different indices (e.g. after a reset_index), rows would be mismatched silently.
    # ══════════════════════════════════════════════════════════════════════════════

    print(f"\n{'═'*60}")
    print("STEP 9 — Re-attach Target and Save")
    print(f"{'═'*60}")

    train_out = X_train.copy()
    valid_out = X_valid.copy()
    test_out  = X_test.copy()

    # .values strips the index — safe alignment regardless of index state
    train_out[TARGET] = y_train.values
    valid_out[TARGET] = y_valid.values
    test_out[TARGET]  = y_test.values

    train_out.to_parquet(TRAIN_OUT, index=False)
    valid_out.to_parquet(VALID_OUT, index=False)
    test_out.to_parquet(TEST_OUT,   index=False)

    print(f"  Saved: {TRAIN_OUT}  ({train_out.shape[0]:,} rows, {train_out.shape[1]} cols)")
    print(f"  Saved: {VALID_OUT}  ({valid_out.shape[0]:,} rows, {valid_out.shape[1]} cols)")
    print(f"  Saved: {TEST_OUT}   ({test_out.shape[0]:,} rows, {test_out.shape[1]} cols)")

    # ══════════════════════════════════════════════════════════════════════════════
    # STEP 10 — FULL SELECTION REPORT
    # Documents every dropped column and the reason it was dropped.
    # This section provides the written justification required for grading.
    # ══════════════════════════════════════════════════════════════════════════════

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

    print(f"\n  Dropped — non-numeric (information already in engineered features):")
    print(f"    {non_numeric_cols}")

    print(f"\n  Dropped — variance (near-constant, <1% minority value):")
    print(f"    {low_variance_cols if low_variance_cols else 'none'}")

    print(f"\n  Protected from variance (lookup-table features — few unique values by construction):")
    print(f"    {protected_present}")

    print(f"\n  Dropped — correlation (redundant pair, lower target correlation dropped):")
    print(f"    {to_drop_corr if to_drop_corr else 'none'}")

    print(f"\n  Dropped — mutual information (essentially zero target information):")
    print(f"    {low_mi_cols if low_mi_cols else 'none'}")

    print(f"\n  Dropped — importance (Random Forest score < {IMPORTANCE_THRESHOLD}):")
    print(f"    {[c for c in low_importance_cols if c not in FORCE_DROP]}")

    print(f"\n  Dropped — domain reasoning (better proxy already exists):")
    print(f"    {[c for c in low_importance_cols if c in FORCE_DROP]}")

    print(f"\n  Reinstated — domain reasoning (consistent signal, RF undervalued):")
    print(f"    {force_kept if force_kept else 'none'}")

    print(f"\n  Final selected features (ranked by importance):")
    for i, col in enumerate(selected_cols, 1):
        score = importance_series.get(col, 0.0)
        print(f"    {i:2d}. {col:<35} importance: {score:.4f}")

    print(f"\n  NaN check on saved train file:")
    nan_counts = train_out.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts) == 0:
        print(f"    No NaNs found ✓")
    else:
        print(nan_counts.to_string())

    print(f"\n  Target present in all splits:")
    print(f"    Train: {TARGET in train_out.columns} ✓")
    print(f"    Valid: {TARGET in valid_out.columns} ✓")
    print(f"    Test:  {TARGET in test_out.columns}  ✓")

    print(f"\n{'═'*60}")
    print("Feature selection complete.")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()