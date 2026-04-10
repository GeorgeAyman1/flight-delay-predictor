from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================
INPUT_PATH = Path(
    r"C:\Users\VICTUS\Desktop\Engineering\Sem 8\Data Science\flight-delay-predictor\data\interim\merged_dataset.parquet"
)

REPO_ROOT = INPUT_PATH.parents[2]

# Save as v2 first so you keep the old outputs for comparison
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "cleaned_merged_dataset_v2.parquet"

TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train_2022_2023_v2.parquet"
VALID_PATH = REPO_ROOT / "data" / "processed" / "valid_2024_v2.parquet"
TEST_PATH = REPO_ROOT / "data" / "processed" / "test_2025_v2.parquet"

REPORT_DIR = REPO_ROOT / "reports" / "tables"
LOG_CSV_PATH = REPORT_DIR / "cleaning_log_v2.csv"
SUMMARY_JSON_PATH = REPORT_DIR / "cleaning_summary_v2.json"
SUMMARY_MD_PATH = REPORT_DIR / "cleaning_summary_v2.md"
SPLIT_SUMMARY_PATH = REPORT_DIR / "dataset_split_summary_v2.json"

ALLOWED_CARRIERS = {"AA", "AS", "B6", "DL", "UA"}
DROP_FLIGHT_KEY_DUPLICATES = False


# =============================================================================
# Helpers
# =============================================================================
def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def add_log(
    logs: List[Dict],
    step_number: int,
    action: str,
    columns: str,
    details: str,
    reason: str,
    rows_affected: int,
) -> None:
    logs.append(
        {
            "step_number": step_number,
            "action": action,
            "columns": columns,
            "details": details,
            "reason": reason,
            "rows_affected": int(rows_affected),
        }
    )


def safe_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    cleaned = cleaned.replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA, "<NA>": pd.NA}
    )
    return pd.to_numeric(cleaned, errors="coerce")


def markdown_table_from_logs(log_df: pd.DataFrame) -> str:
    if log_df.empty:
        return "_No logged cleaning actions._"

    header = (
        "| Step | Action | Columns | Rows Affected | Details | Reason |\n"
        "|---:|---|---|---:|---|---|"
    )

    rows = []
    for _, r in log_df.iterrows():
        rows.append(
            f"| {r['step_number']} | "
            f"{str(r['action']).replace('|', '/')} | "
            f"{str(r['columns']).replace('|', '/')} | "
            f"{int(r['rows_affected'])} | "
            f"{str(r['details']).replace('|', '/')} | "
            f"{str(r['reason']).replace('|', '/')} |"
        )

    return "\n".join([header] + rows)


def write_markdown_summary(
    summary_path: Path,
    input_path: Path,
    output_path: Path,
    before_shape: Tuple[int, int],
    after_shape: Tuple[int, int],
    removed_rows: int,
    log_df: pd.DataFrame,
    missing_before: pd.Series,
    missing_after: pd.Series,
    dtype_before: Dict[str, str],
    dtype_after: Dict[str, str],
    class_distribution: Optional[Dict],
    sparse_missing_counts: Dict[str, int],
    split_summary: Dict,
) -> None:
    report_lines = [
        "# Cleaning Summary",
        "",
        f"**Input file:** `{input_path}`",
        f"**Output file:** `{output_path}`",
        "",
        "## Shape",
        "",
        f"- Before: **{before_shape[0]:,} rows × {before_shape[1]} columns**",
        f"- After: **{after_shape[0]:,} rows × {after_shape[1]} columns**",
        f"- Rows removed: **{removed_rows:,}**",
        "",
        "## Logged Cleaning Steps",
        "",
        markdown_table_from_logs(log_df),
        "",
        "## Missing Values (Top 20 Before)",
        "",
        "```",
        missing_before.sort_values(ascending=False).head(20).to_string(),
        "```",
        "",
        "## Missing Values (Top 20 After)",
        "",
        "```",
        missing_after.sort_values(ascending=False).head(20).to_string(),
        "```",
        "",
        "## Sparse Weather Fields Left As-Is",
        "",
        "These were not treated as ordinary errors because their missingness can be meaningful.",
        "",
        "```",
        pd.Series(sparse_missing_counts).to_string() if sparse_missing_counts else "No sparse weather columns found.",
        "```",
        "",
        "## Data Types Before",
        "",
        "```",
        pd.Series(dtype_before).astype(str).to_string(),
        "```",
        "",
        "## Data Types After",
        "",
        "```",
        pd.Series(dtype_after).astype(str).to_string(),
        "```",
        "",
        "## Temporal Split Summary",
        "",
        "```",
        pd.Series(
            {
                "train_rows": split_summary["train_shape"]["rows"],
                "validation_rows": split_summary["validation_shape"]["rows"],
                "test_rows": split_summary["test_shape"]["rows"],
            }
        ).to_string(),
        "```",
    ]

    if class_distribution:
        report_lines.extend(
            [
                "",
                "## Target Distribution",
                "",
                "```",
                pd.Series(class_distribution).to_string(),
                "```",
            ]
        )

    if "class_distribution" in split_summary:
        report_lines.extend(
            [
                "",
                "## Split Target Distribution",
                "",
                "```",
                json.dumps(split_summary["class_distribution"], indent=2),
                "```",
            ]
        )

    summary_path.write_text("\n".join(report_lines), encoding="utf-8")


def get_split_year_series(df: pd.DataFrame) -> pd.Series:
    preferred_datetime_cols = [
        "scheduled_departure_dt",
        "date_dt",
        "date_mmddyyyy",
        "valid",
    ]
    for col in preferred_datetime_cols:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            return df[col].dt.year.astype("Int64")

    fallback_year_cols = ["year_x", "year_y", "year"]
    for col in fallback_year_cols:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype("Int64")

    raise ValueError("Could not determine split year from datetime/year columns.")


def split_dataset_by_year(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    split_year = get_split_year_series(df)

    train_df = df.loc[split_year.isin([2022, 2023])].copy()
    valid_df = df.loc[split_year == 2024].copy()
    test_df = df.loc[split_year == 2025].copy()

    summary = {
        "train_years": [2022, 2023],
        "validation_years": [2024],
        "test_years": [2025],
        "train_shape": {"rows": int(train_df.shape[0]), "cols": int(train_df.shape[1])},
        "validation_shape": {"rows": int(valid_df.shape[0]), "cols": int(valid_df.shape[1])},
        "test_shape": {"rows": int(test_df.shape[0]), "cols": int(test_df.shape[1])},
    }

    if "departure_delayed" in df.columns:
        summary["class_distribution"] = {
            "train": {
                str(k): int(v)
                for k, v in train_df["departure_delayed"].value_counts(dropna=False).items()
            },
            "validation": {
                str(k): int(v)
                for k, v in valid_df["departure_delayed"].value_counts(dropna=False).items()
            },
            "test": {
                str(k): int(v)
                for k, v in test_df["departure_delayed"].value_counts(dropna=False).items()
            },
        }

    return train_df, valid_df, test_df, summary


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)

    logs: List[Dict] = []
    step = 1

    # -------------------------------------------------------------------------
    # Column discovery
    # -------------------------------------------------------------------------
    carrier_col = find_col(df, ["carrier_code", "carrier", "op_unique_carrier", "marketing_carrier"])
    origin_col = find_col(df, ["origin_airport", "origin", "origin_code"])
    dest_col = find_col(df, ["destination_airport", "dest_airport", "dest", "destination"])
    flight_num_col = find_col(df, ["flight_number", "fl_num", "flight_num"])

    flight_date_col = find_col(df, ["flight_date", "fl_date", "date_dt", "date_mmddyyyy", "date"])
    sched_dep_dt_col = find_col(df, ["scheduled_departure_dt", "sched_dep_dt"])
    valid_col = find_col(df, ["valid", "weather_timestamp", "observation_time", "obs_time"])

    dep_delay_col = find_col(df, ["departure_delay_minutes", "dep_delay_minutes", "dep_delay"])

    # Fields that should remain as strings / codes, not datetimes
    hhmm_time_cols = [
        c
        for c in [
            find_col(df, ["scheduled_departure_time", "crs_dep_time"]),
            find_col(df, ["actual_departure_time", "dep_time"]),
            find_col(df, ["wheelsoff_time", "wheels_off"]),
            find_col(df, ["timezone"]),
        ]
        if c is not None
    ]

    temp_col = find_col(df, ["tmpf", "temperature", "temp_f", "air_temperature"])
    dew_col = find_col(df, ["dwpf", "dew_point", "dewpoint", "dew_point_f"])
    wind_col = find_col(df, ["sknt", "wind_speed", "wind_speed_kt", "wind_speed_knots"])
    gust_col = find_col(df, ["gust", "gust_speed", "wind_gust"])
    humidity_col = find_col(df, ["relh", "humidity", "relative_humidity"])
    visibility_col = find_col(df, ["vsby", "visibility", "visibility_miles"])
    precip_col = find_col(df, ["p01i", "precipitation", "precip", "precip_in"])
    pressure_col = find_col(df, ["alti", "mslp", "pressure", "sea_level_pressure"])

    sparse_weather_cols = [
        c
        for c in [
            gust_col,
            find_col(df, ["wxcodes", "weather_codes", "weather_code"]),
            find_col(df, ["skyc2"]),
            find_col(df, ["skyc3"]),
            find_col(df, ["skyc4"]),
            find_col(df, ["skyl2"]),
            find_col(df, ["skyl3"]),
            find_col(df, ["skyl4"]),
        ]
        if c is not None
    ]

    shape_before = df.shape
    missing_before = df.isna().sum()
    dtype_before = df.dtypes.astype(str).to_dict()

    # -------------------------------------------------------------------------
    # 1) Trim whitespace in text columns
    # -------------------------------------------------------------------------
    obj_cols = list(df.select_dtypes(include=["object", "string"]).columns)
    changed_cells = 0
    for col in obj_cols:
        before_col = df[col].astype("string")
        after_col = before_col.str.strip()
        changed_cells += int((before_col.fillna("<NA>") != after_col.fillna("<NA>")).sum())
        df[col] = after_col

    add_log(
        logs,
        step,
        "Standardization",
        "all object/string columns",
        "Trimmed leading/trailing whitespace from text columns.",
        "Whitespace can break joins, filters, grouping, and categorical consistency.",
        changed_cells,
    )
    step += 1

    # -------------------------------------------------------------------------
    # 2) Uppercase carrier/airport code columns
    # -------------------------------------------------------------------------
    code_like_cols = [c for c in [carrier_col, origin_col, dest_col] if c is not None]
    code_changes = 0
    for col in code_like_cols:
        before_col = df[col].astype("string")
        after_col = before_col.str.upper()
        code_changes += int((before_col.fillna("<NA>") != after_col.fillna("<NA>")).sum())
        df[col] = after_col

    if code_like_cols:
        add_log(
            logs,
            step,
            "Standardization",
            ", ".join(code_like_cols),
            "Uppercased carrier and airport codes.",
            "Codes should be case-consistent before validation and modeling.",
            code_changes,
        )
        step += 1

    # -------------------------------------------------------------------------
    # 3) Convert only confirmed datetime columns
    # -------------------------------------------------------------------------
    explicit_datetime_candidates = [flight_date_col, sched_dep_dt_col, valid_col]
    datetime_candidates = [c for c in explicit_datetime_candidates if c is not None]
    datetime_candidates = list(dict.fromkeys(datetime_candidates))

    dt_changed = 0
    for col in datetime_candidates:
        before_col = df[col].astype("string")
        converted = pd.to_datetime(df[col], errors="coerce")
        dt_changed += int((before_col.fillna("<NA>") != converted.astype("string").fillna("<NA>")).sum())
        df[col] = converted

    if datetime_candidates:
        add_log(
            logs,
            step,
            "Type coercion",
            ", ".join(datetime_candidates),
            "Converted confirmed date/datetime columns to pandas datetime.",
            "Only true timestamps were parsed. HHMM operational time fields and timezone were left untouched.",
            dt_changed,
        )
        step += 1

    # -------------------------------------------------------------------------
    # 4) Preserve HHMM-like time fields and timezone as clean strings
    # -------------------------------------------------------------------------
    preserved_string_changes = 0
    for col in hhmm_time_cols:
        before_col = df[col].astype("string")
        after_col = before_col.str.strip()
        preserved_string_changes += int((before_col.fillna("<NA>") != after_col.fillna("<NA>")).sum())
        df[col] = after_col

    if hhmm_time_cols:
        add_log(
            logs,
            step,
            "Preservation",
            ", ".join(hhmm_time_cols),
            "Kept operational HHMM-style time fields and timezone as strings instead of parsing them as datetimes.",
            "These are identifiers/clock fields, not full timestamps.",
            preserved_string_changes,
        )
        step += 1

    # -------------------------------------------------------------------------
    # 5) Convert numeric columns
    # -------------------------------------------------------------------------
    numeric_candidates = [
        c
        for c in [
            dep_delay_col,
            temp_col,
            dew_col,
            wind_col,
            gust_col,
            humidity_col,
            visibility_col,
            precip_col,
            pressure_col,
        ]
        if c is not None
    ]

    exclusion_cols = set(datetime_candidates + hhmm_time_cols)

    for col in df.columns:
        name = col.lower()
        if col in exclusion_cols:
            continue

        if any(
            token in name
            for token in [
                "delay",
                "minutes",
                "distance",
                "air_time",
                "elapsed_time",
                "taxi_out",
                "taxiout",
                "taxi_in",
                "flight_number",
                "latitude",
                "longitude",
                "elevation",
                "runways",
                "year_",
                "year",
                "lon",
                "lat",
            ]
        ):
            numeric_candidates.append(col)

    numeric_candidates = list(dict.fromkeys(numeric_candidates))
    numeric_changes = 0
    for col in numeric_candidates:
        before_col = df[col].astype("string")
        converted = safe_numeric(df[col])
        numeric_changes += int((before_col.fillna("<NA>") != converted.astype("string").fillna("<NA>")).sum())
        df[col] = converted

    if numeric_candidates:
        add_log(
            logs,
            step,
            "Type coercion",
            ", ".join(numeric_candidates),
            "Converted likely numeric columns to numeric dtype using coercion.",
            "Numeric consistency is required for anomaly checks, summaries, and modeling.",
            numeric_changes,
        )
        step += 1

    # -------------------------------------------------------------------------
    # 6) Remove exact duplicates
    # -------------------------------------------------------------------------
    exact_dup_count = int(df.duplicated().sum())
    if exact_dup_count > 0:
        df = df.drop_duplicates().copy()

    add_log(
        logs,
        step,
        "Removal",
        "all columns",
        "Removed exact duplicate rows." if exact_dup_count > 0 else "No exact duplicate rows found.",
        "Exact duplicates add no information and can bias analysis.",
        exact_dup_count,
    )
    step += 1

    # -------------------------------------------------------------------------
    # 7) Audit business-key duplicates
    # -------------------------------------------------------------------------
    flight_key_cols = [c for c in [flight_date_col, carrier_col, flight_num_col, origin_col, dest_col, sched_dep_dt_col] if c is not None]
    key_dup_count = 0
    if DROP_FLIGHT_KEY_DUPLICATES and len(flight_key_cols) >= 4:
        key_dup_count = int(df.duplicated(subset=flight_key_cols).sum())
        if key_dup_count > 0:
            df = df.drop_duplicates(subset=flight_key_cols, keep="first").copy()

    add_log(
        logs,
        step,
        "Audit" if not DROP_FLIGHT_KEY_DUPLICATES else "Removal",
        ", ".join(flight_key_cols) if flight_key_cols else "n/a",
        (
            "Checked business-key duplicates only; no rows dropped because DROP_FLIGHT_KEY_DUPLICATES=False."
            if not DROP_FLIGHT_KEY_DUPLICATES
            else "Dropped duplicated rows based on the flight business key."
        ),
        "Business-key duplicate removal should be explicit, not accidental.",
        key_dup_count,
    )
    step += 1

    # -------------------------------------------------------------------------
    # 8) Remove known junk/invalid carrier rows
    # -------------------------------------------------------------------------
    removed_junk_rows = 0
    removed_invalid_carrier_rows = 0
    if carrier_col is not None:
        carrier_series = df[carrier_col].astype("string")

        junk_mask = carrier_series.str.contains(
            "SOURCE: BUREAU OF TRANSPORTATION STATISTICS",
            case=False,
            na=False,
        )
        removed_junk_rows = int(junk_mask.sum())
        if removed_junk_rows > 0:
            df = df.loc[~junk_mask].copy()

        invalid_carrier_mask = ~df[carrier_col].isin(ALLOWED_CARRIERS)
        removed_invalid_carrier_rows = int(invalid_carrier_mask.sum())
        if removed_invalid_carrier_rows > 0:
            df = df.loc[~invalid_carrier_mask].copy()

    add_log(
        logs,
        step,
        "Removal",
        carrier_col if carrier_col else "carrier column not found",
        "Removed the known BTS source/junk row and carrier codes outside the 5 project carriers.",
        "Keeps the dataset aligned with the documented project scope.",
        removed_junk_rows + removed_invalid_carrier_rows,
    )
    step += 1

    # -------------------------------------------------------------------------
    # 9) Weather anomaly: dew point > temperature
    # -------------------------------------------------------------------------
    if dew_col and temp_col:
        mask = df[dew_col].notna() & df[temp_col].notna() & (df[dew_col] > df[temp_col])
        count = int(mask.sum())
        if count > 0:
            df.loc[mask, dew_col] = pd.NA

        add_log(
            logs,
            step,
            "Cell nullification",
            dew_col,
            f"Set `{dew_col}` to missing where `{dew_col} > {temp_col}`.",
            "Dew point above air temperature indicates a bad sensor reading.",
            count,
        )
        step += 1

    # -------------------------------------------------------------------------
    # 10) Weather anomaly: gust < sustained wind
    # -------------------------------------------------------------------------
    if gust_col and wind_col:
        mask = df[gust_col].notna() & df[wind_col].notna() & (df[gust_col] < df[wind_col])
        count = int(mask.sum())
        if count > 0:
            df.loc[mask, gust_col] = pd.NA

        add_log(
            logs,
            step,
            "Cell nullification",
            gust_col,
            f"Set `{gust_col}` to missing where `{gust_col} < {wind_col}`.",
            "Wind gust lower than sustained wind indicates a bad sensor reading.",
            count,
        )
        step += 1

    # -------------------------------------------------------------------------
    # 11) Impossible-range checks
    # -------------------------------------------------------------------------
    def nullify_invalid(mask: pd.Series, col_name: str, description: str) -> None:
        nonlocal step
        count = int(mask.sum())
        if count > 0:
            df.loc[mask, col_name] = pd.NA

        add_log(
            logs,
            step,
            "Cell nullification",
            col_name,
            description,
            "Clearly impossible physical values should not be retained as valid measurements.",
            count,
        )
        step += 1

    if humidity_col:
        nullify_invalid(
            df[humidity_col].notna() & ((df[humidity_col] < 0) | (df[humidity_col] > 100)),
            humidity_col,
            f"Set `{humidity_col}` to missing where values were outside [0, 100].",
        )

    if wind_col:
        nullify_invalid(
            df[wind_col].notna() & (df[wind_col] < 0),
            wind_col,
            f"Set `{wind_col}` to missing where values were negative.",
        )

    if gust_col:
        nullify_invalid(
            df[gust_col].notna() & (df[gust_col] < 0),
            gust_col,
            f"Set `{gust_col}` to missing where values were negative.",
        )

    if visibility_col:
        nullify_invalid(
            df[visibility_col].notna() & (df[visibility_col] < 0),
            visibility_col,
            f"Set `{visibility_col}` to missing where values were negative.",
        )

    if precip_col:
        nullify_invalid(
            df[precip_col].notna() & (df[precip_col] < 0),
            precip_col,
            f"Set `{precip_col}` to missing where values were negative.",
        )

    # -------------------------------------------------------------------------
    # 12) Create / standardize target
    # -------------------------------------------------------------------------
    if dep_delay_col is None:
        raise ValueError("Could not find `departure_delay_minutes` column to create the target.")

    df["departure_delayed"] = (df[dep_delay_col] >= 15).astype("Int8")
    add_log(
        logs,
        step,
        "Feature creation",
        "departure_delayed",
        f"Created the binary target from `{dep_delay_col}` using threshold >= 15 minutes.",
        "This is the project target used for classification.",
        len(df),
    )
    step += 1

    # -------------------------------------------------------------------------
    # 13) Preserve sparse weather fields as-is
    # -------------------------------------------------------------------------
    sparse_missing_counts = {col: int(df[col].isna().sum()) for col in sparse_weather_cols}
    if sparse_weather_cols:
        add_log(
            logs,
            step,
            "Documentation",
            ", ".join(sparse_weather_cols),
            "Preserved natural sparsity in sparse weather fields; no blanket imputation was performed here.",
            "These columns are sparse by nature, so missingness is meaningful.",
            sum(sparse_missing_counts.values()),
        )
        step += 1

    # -------------------------------------------------------------------------
    # 14) Temporal split
    # -------------------------------------------------------------------------
    train_df, valid_df, test_df, split_summary = split_dataset_by_year(df)

    train_df.to_parquet(TRAIN_PATH, index=False)
    valid_df.to_parquet(VALID_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)

    SPLIT_SUMMARY_PATH.write_text(
        json.dumps(split_summary, indent=2, default=str),
        encoding="utf-8",
    )

    add_log(
        logs,
        step,
        "Dataset split",
        "year-based temporal split",
        "Split cleaned dataset into train (2022-2023), validation (2024), and test (2025).",
        "Prevents temporal leakage and reflects real-world forecasting conditions.",
        len(df),
    )
    step += 1

    # -------------------------------------------------------------------------
    # Save cleaned full dataset and reports
    # -------------------------------------------------------------------------
    shape_after = df.shape
    missing_after = df.isna().sum()
    dtype_after = df.dtypes.astype(str).to_dict()
    removed_rows = shape_before[0] - shape_after[0]

    class_distribution = (
        df["departure_delayed"]
        .value_counts(dropna=False)
        .rename(index={0: "on_time", 1: "delayed"})
        .to_dict()
    )

    df.to_parquet(OUTPUT_PATH, index=False)

    log_df = pd.DataFrame(logs).sort_values("step_number")
    log_df.to_csv(LOG_CSV_PATH, index=False)

    summary_payload = {
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "shape_before": {"rows": int(shape_before[0]), "cols": int(shape_before[1])},
        "shape_after": {"rows": int(shape_after[0]), "cols": int(shape_after[1])},
        "rows_removed": int(removed_rows),
        "steps_logged": int(len(log_df)),
        "class_distribution": class_distribution,
        "sparse_weather_missing_counts": sparse_missing_counts,
        "top_missing_before": missing_before.sort_values(ascending=False).head(20).to_dict(),
        "top_missing_after": missing_after.sort_values(ascending=False).head(20).to_dict(),
        "split_summary": split_summary,
    }
    SUMMARY_JSON_PATH.write_text(
        json.dumps(summary_payload, indent=2, default=str),
        encoding="utf-8",
    )

    write_markdown_summary(
        summary_path=SUMMARY_MD_PATH,
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        before_shape=shape_before,
        after_shape=shape_after,
        removed_rows=removed_rows,
        log_df=log_df,
        missing_before=missing_before,
        missing_after=missing_after,
        dtype_before=dtype_before,
        dtype_after=dtype_after,
        class_distribution=class_distribution,
        sparse_missing_counts=sparse_missing_counts,
        split_summary=split_summary,
    )

    print("=" * 88)
    print("Cleaning + temporal split completed successfully.")
    print(f"Input file : {INPUT_PATH}")
    print(f"Cleaned full dataset : {OUTPUT_PATH}")
    print(f"Train set  : {TRAIN_PATH}")
    print(f"Valid set  : {VALID_PATH}")
    print(f"Test set   : {TEST_PATH}")
    print(f"Before     : {shape_before[0]:,} rows x {shape_before[1]} columns")
    print(f"After      : {shape_after[0]:,} rows x {shape_after[1]} columns")
    print(f"Rows removed : {removed_rows:,}")
    print(f"Log CSV    : {LOG_CSV_PATH}")
    print(f"Summary JSON : {SUMMARY_JSON_PATH}")
    print(f"Summary MD : {SUMMARY_MD_PATH}")
    print(f"Split JSON : {SPLIT_SUMMARY_PATH}")
    print("=" * 88)


if __name__ == "__main__":
    main()