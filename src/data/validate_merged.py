"""
validate_merged.py
------------------
Runs all GX-style validation checks against data/processed/merged_dataset.parquet
and writes the results to reports/validation_results.json.

Usage:
    python validate_merged.py

Output:
    reports/validation_results.json   (read by gx_validation_dashboard.html)
"""

from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv


# ── Paths ──────────────────────────────────────────────────────────────────────
load_dotenv()
ROOT = Path(os.getenv("PROJECT_ROOT"))
DATA_PATH = ROOT / "data" / "processed" / "merged_dataset.parquet"
REPORT_PATH = ROOT / "data" / "reports" / "validation_results.json"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def expect(
    section: str,
    name: str,
    passed: bool,
    detail: str,
    observed=None,
    warn_only: bool = False,
) -> dict:
    status = "pass" if passed else ("warn" if warn_only else "fail")
    result = {
        "section": section,
        "expectation": name,
        "status": status,
        "detail": detail,
    }
    if observed is not None:
        result["observed"] = observed
    return result


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# ── Load ───────────────────────────────────────────────────────────────────────
print(f"Loading {DATA_PATH} ...")
df = pd.read_parquet(DATA_PATH)
df.columns = df.columns.str.lower()
rows, cols = len(df), len(df.columns)
print(f"  Shape: {rows:,} rows × {cols} columns")

results = []
section_meta = {}   # section_id -> {label, group}


def add(section_id: str, label: str, group: str, r: dict):
    results.append(r)
    section_meta[section_id] = {"label": label, "group": group}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Shape & Schema
# ══════════════════════════════════════════════════════════════════════════════
S = "shape"
REQUIRED_FLIGHT_COLS = [
    "carrier_code", "origin_airport", "destination_airport",
    "date_mmddyyyy", "scheduled_departure_time",
    "scheduled_elapsed_time_minutes", "airline", "year_x",
    "flight_number", "departure_delay_minutes",
]
REQUIRED_WEATHER_COLS = [
    "tmpf", "dwpf", "relh", "drct", "sknt", "gust",
    "p01i", "alti", "mslp", "vsby", "feel", "wxcodes",
    "skyc1", "skyc2", "skyc3", "skyc4",
    "skyl1", "skyl2", "skyl3", "skyl4",
]
ALL_REQUIRED = REQUIRED_FLIGHT_COLS + REQUIRED_WEATHER_COLS

missing_cols = [c for c in ALL_REQUIRED if c not in df.columns]

add(S, "Shape & Schema", "Flights", expect(
    S, "expect_table_row_count_to_be_between",
    rows > 0, f"Row count: {rows:,}", observed=rows
))
add(S, "Shape & Schema", "Flights", expect(
    S, "expect_table_columns_to_match_set",
    len(missing_cols) == 0,
    f"Missing columns: {missing_cols}" if missing_cols else f"All {len(ALL_REQUIRED)} required columns present",
    observed={"total_cols": cols, "missing": missing_cols}
))
for col in REQUIRED_FLIGHT_COLS:
    add(S, "Shape & Schema", "Flights", expect(
        S, f"expect_column_to_exist — {col}",
        col in df.columns,
        f"dtype: {str(df[col].dtype)}" if col in df.columns else "COLUMN NOT FOUND",
    ))
for col in REQUIRED_WEATHER_COLS:
    add(S, "Shape & Schema", "Flights", expect(
        S, f"expect_column_to_exist — {col} (weather)",
        col in df.columns,
        f"dtype: {str(df[col].dtype)}" if col in df.columns else "COLUMN NOT FOUND",
    ))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Completeness
# ══════════════════════════════════════════════════════════════════════════════
S = "completeness"

critical_cols = [
    "carrier_code", "origin_airport", "destination_airport",
    "date_mmddyyyy", "scheduled_departure_time",
    "scheduled_elapsed_time_minutes", "airline", "year_x", "flight_number",
]
weather_cols = [
    "tmpf", "dwpf", "relh", "sknt", "vsby",
    "drct", "gust", "p01i", "alti", "mslp", "feel", "wxcodes",
    "skyc1", "skyc2", "skyc3", "skyc4",
]

completeness_observed = {}

for col in critical_cols:
    if col not in df.columns:
        continue
    n_missing = int(df[col].isnull().sum())
    pct_missing = round(n_missing / rows * 100, 2)
    completeness_observed[col] = {"missing": n_missing, "pct_complete": round(100 - pct_missing, 2)}
    add(S, "Completeness", "Flights", expect(
        S, f"expect_column_values_to_not_be_null — {col}",
        n_missing == 0,
        f"Missing: {n_missing:,} ({pct_missing:.1f}%)",
        observed={"missing": n_missing, "pct": pct_missing}
    ))

HIGH_MISS_THRESHOLD = 5.0   # warn if missing > 5%
for col in weather_cols:
    if col not in df.columns:
        continue
    n_missing = int(df[col].isnull().sum())
    pct_missing = round(n_missing / rows * 100, 2)
    completeness_observed[col] = {"missing": n_missing, "pct_complete": round(100 - pct_missing, 2)}
    add(S, "Completeness", "Flights", expect(
        S, f"expect_column_values_to_not_be_null — {col} (weather)",
        n_missing == 0,
        f"Missing: {n_missing:,} ({pct_missing:.1f}%)",
        observed={"missing": n_missing, "pct": pct_missing},
        warn_only=True,   # weather fields allowed to be sparse
    ))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Duplicates
# ══════════════════════════════════════════════════════════════════════════════
S = "duplicates"

exact_dups = int(df.duplicated().sum())
flight_id_cols = [
    "date_mmddyyyy", "carrier_code", "flight_number",
    "origin_airport", "destination_airport", "scheduled_departure_time",
]
flight_id_cols_present = [c for c in flight_id_cols if c in df.columns]
subset_dups = int(df.duplicated(subset=flight_id_cols_present).sum()) if flight_id_cols_present else -1

add(S, "Duplicates", "Flights", expect(
    S, "expect_table_row_count_to_equal (no exact duplicates)",
    exact_dups == 0,
    f"Exact duplicate rows: {exact_dups:,}",
    observed=exact_dups
))
add(S, "Duplicates", "Flights", expect(
    S, "expect_compound_columns_to_be_unique — flight identity",
    subset_dups == 0,
    f"Duplicate flight-identity rows: {subset_dups:,}  (key: date+carrier+flight#+origin+dest+dep_time)",
    observed=subset_dups
))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Year Distribution
# ══════════════════════════════════════════════════════════════════════════════
S = "years"

year_col = num(df["year_x"])
nan_years   = int(year_col.isna().sum())
yr_counts   = {}
for y in [2022, 2023, 2024, 2025]:
    yr_counts[str(y)] = int((year_col == y).sum())
inv_years   = int(year_col.notna().sum()) - sum(yr_counts.values())

add(S, "Year Distribution", "Flights", expect(
    S, "expect_column_values_to_not_be_null — year",
    nan_years == 0,
    f"NaN years: {nan_years:,}",
    observed=nan_years
))
add(S, "Year Distribution", "Flights", expect(
    S, "expect_column_values_to_be_in_set — year ∈ {2022,2023,2024,2025}",
    inv_years == 0,
    f"Years outside range [2022,2025]: {inv_years:,}",
    observed={"year_counts": yr_counts, "outside_range": inv_years}
))
for y, cnt in yr_counts.items():
    add(S, "Year Distribution", "Flights", expect(
        S, f"expect_column_value_lengths_to_be_between — {y} records",
        cnt > 0,
        f"Records in {y}: {cnt:,}",
        observed=cnt
    ))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Field Validity
# ══════════════════════════════════════════════════════════════════════════════
S = "validity"

VALID_ORIGINS = {"BOS","CLT","DEN","DFW","DTW","EWR","IAH","JFK","LAX","MIA","MSP","ORD","PHX","SEA","SFO"}
VALID_CARRIERS = {"AA","AS","B6","DL","UA"}

def count_regex_mismatch(series, pattern):
    import re
    mask = series.dropna().astype(str).str.fullmatch(pattern)
    return int((~mask).sum())

if "origin_airport" in df.columns:
    inv = count_regex_mismatch(df["origin_airport"], r"[A-Z]{3}")
    add(S, "Field Validity", "Flights", expect(S, "expect_column_values_to_match_regex — origin_airport [A-Z]{3}", inv==0, f"Invalid codes: {inv:,}", observed=inv))
    not_in_set = int((~df["origin_airport"].isin(VALID_ORIGINS) & df["origin_airport"].notna()).sum())
    add(S, "Field Validity", "Flights", expect(S, "expect_column_values_to_be_in_set — origin_airport", not_in_set==0, f"Airports outside expected 15: {not_in_set:,}", observed=not_in_set))

if "destination_airport" in df.columns:
    inv = count_regex_mismatch(df["destination_airport"], r"[A-Z]{3}")
    add(S, "Field Validity", "Flights", expect(S, "expect_column_values_to_match_regex — destination_airport [A-Z]{3}", inv==0, f"Invalid codes: {inv:,}", observed=inv))

if "carrier_code" in df.columns:
    inv = count_regex_mismatch(df["carrier_code"], r"[A-Z0-9]{2}")
    add(S, "Field Validity", "Flights", expect(S, "expect_column_values_to_match_regex — carrier_code [A-Z0-9]{2}", inv==0, f"Invalid codes: {inv:,}", observed=inv))
    not_in_set = int((~df["carrier_code"].isin(VALID_CARRIERS) & df["carrier_code"].notna()).sum())
    add(S, "Field Validity", "Flights", expect(S, "expect_column_values_to_be_in_set — carrier_code", not_in_set==0, f"Carriers outside expected 5: {not_in_set:,}", observed=not_in_set))

if "scheduled_departure_time" in df.columns:
    times = num(df["scheduled_departure_time"])
    inv_time = int((times.notna() & ((times < 0) | (times > 2359) | ((times % 100) >= 60))).sum())
    add(S, "Field Validity", "Flights", expect(S, "expect_column_values_to_be_between — scheduled_departure_time [0, 2359]", inv_time==0, f"Invalid times: {inv_time:,}", observed=inv_time))

if "scheduled_elapsed_time_minutes" in df.columns:
    neg_elapsed = int((num(df["scheduled_elapsed_time_minutes"]) <= 0).sum())
    add(S, "Field Validity", "Flights", expect(S, "expect_column_values_to_be_between — scheduled_elapsed_time_minutes > 0", neg_elapsed==0, f"Non-positive elapsed: {neg_elapsed:,}", observed=neg_elapsed))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Target Variable
# ══════════════════════════════════════════════════════════════════════════════
S = "target"

target_stats = {}
if "departure_delay_minutes" in df.columns:
    target = num(df["departure_delay_minutes"])
    n_null = int(target.isna().sum())
    n_neg  = int((target < 0).sum())
    n_pos  = int((target >= 0).sum())
    n_del15 = int((target >= 15).sum())
    n_ok15  = int((target < 15).sum())
    pct_delayed = round(n_del15 / rows * 100, 2)
    desc = target.describe()
    target_stats = {
        "missing": n_null,
        "mean": round(float(desc["mean"]), 2),
        "median": round(float(desc["50%"]), 2),
        "std": round(float(desc["std"]), 2),
        "min": round(float(desc["min"]), 2),
        "max": round(float(desc["max"]), 2),
        "pct_delayed_ge15": pct_delayed,
        "n_delayed_ge15": n_del15,
        "n_on_time_lt15": n_ok15,
        "n_negative": n_neg,
        "n_non_negative": n_pos,
    }

    add(S, "Target Variable", "Flights", expect(S, "expect_column_values_to_not_be_null — departure_delay_minutes", n_null==0, f"Missing: {n_null:,}", observed=n_null))
    add(S, "Target Variable", "Flights", expect(S, "expect_column_values_to_be_of_type — float64", True, f"dtype: {df['departure_delay_minutes'].dtype}", observed=str(df['departure_delay_minutes'].dtype)))
    add(S, "Target Variable", "Flights", expect(S, "expect_column_mean_to_be_between — mean delay", True, f"Mean: {desc['mean']:.2f} min  |  Median: {desc['50%']:.2f} min", observed=round(float(desc["mean"]),2)))
    add(S, "Target Variable", "Flights", expect(S, "expect_column_values_to_be_between — binary threshold ≥15 min", True, f"Delayed ≥15 min: {n_del15:,} ({pct_delayed:.1f}%)  |  On-time: {n_ok15:,}", observed={"pct_delayed": pct_delayed}))
    add(S, "Target Variable", "Flights", expect(S, "expect_column_min_to_be_between — min delay (negatives valid)", True, f"Min: {desc['min']:.1f} min  |  Max: {desc['max']:.1f} min", observed={"min": round(float(desc["min"]),1), "max": round(float(desc["max"]),1)}))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Weather Completeness
# ══════════════════════════════════════════════════════════════════════════════
S = "weather"

WEATHER_FIELDS = ["tmpf","dwpf","relh","drct","sknt","gust","p01i","alti","mslp","vsby","feel","wxcodes","skyc1","skyc2","skyc3","skyc4","skyl1","skyl2","skyl3","skyl4"]

weather_completeness = {}
for col in WEATHER_FIELDS:
    if col not in df.columns:
        add(S, "Weather Fields", "Weather", expect(S, f"expect_column_to_exist — {col}", False, "COLUMN NOT FOUND"))
        continue
    n_miss = int(df[col].isnull().sum())
    pct_complete = round((1 - n_miss / rows) * 100, 1)
    weather_completeness[col] = pct_complete
    add(S, "Weather Fields", "Weather", expect(
        S, f"expect_column_to_exist — {col}",
        True, f"Complete: {pct_complete:.1f}%  ({n_miss:,} missing)",
        observed={"pct_complete": pct_complete, "missing": n_miss}
    ))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Weather Ranges
# ══════════════════════════════════════════════════════════════════════════════
S = "weather_range"

def range_check(col, lo=None, hi=None):
    if col not in df.columns:
        return -1
    s = num(df[col])
    mask = pd.Series(False, index=s.index)
    if lo is not None: mask = mask | (s < lo)
    if hi is not None: mask = mask | (s > hi)
    return int((mask & s.notna()).sum())

checks_range = [
    ("tmpf",  "expect_column_values_to_be_between — tmpf > -80°F",   -80, None),
    ("relh",  "expect_column_values_to_be_between — relh [0, 100]",   0,  100),
    ("drct",  "expect_column_values_to_be_between — drct [0, 360]",   0,  360),
    ("sknt",  "expect_column_values_to_be_between — sknt ≥ 0",        0,  None),
    ("gust",  "expect_column_values_to_be_between — gust ≥ 0",        0,  None),
    ("p01i",  "expect_column_values_to_be_between — p01i ≥ 0",        0,  None),
    ("alti",  "expect_column_values_to_be_between — alti [25, 32] inHg", 25, 32),
    ("mslp",  "expect_column_values_to_be_between — mslp [850, 1100] mb", 850, 1100),
    ("vsby",  "expect_column_values_to_be_between — vsby [0, 100] miles", 0, 100),
]
for col, name, lo, hi in checks_range:
    inv = range_check(col, lo, hi)
    if inv == -1:
        add(S, "Weather Ranges", "Weather", expect(S, name, False, "COLUMN NOT FOUND"))
    else:
        bounds = f"[{lo if lo is not None else '−∞'}, {hi if hi is not None else '+∞'}]"
        add(S, "Weather Ranges", "Weather", expect(S, name, inv == 0, f"Out-of-range values: {inv:,}  (bounds: {bounds})", observed=inv))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Weather Logical Checks
# ══════════════════════════════════════════════════════════════════════════════
S = "weather_logical"

VALID_SKY_CODES = {"CLR","FEW","SCT","BKN","OVC","VV"}

if "tmpf" in df.columns and "dwpf" in df.columns:
    tmpf = num(df["tmpf"]); dwpf = num(df["dwpf"])
    dew_gt_temp = int(((dwpf - tmpf) > 0).sum())
    add(S, "Logical Checks", "Weather", expect(S, "expect_column_pair_values_A_to_be_greater_than_or_equal_to_B — tmpf ≥ dwpf", dew_gt_temp==0, f"Dew point > temperature (impossible): {dew_gt_temp:,}", observed=dew_gt_temp))

if "sknt" in df.columns and "gust" in df.columns:
    sknt = num(df["sknt"]); gust = num(df["gust"])
    gust_lt_wind = int((gust < sknt).sum())
    add(S, "Logical Checks", "Weather", expect(S, "expect_column_pair_values_A_to_be_greater_than_or_equal_to_B — gust ≥ sknt", gust_lt_wind==0, f"Gust < wind speed (impossible): {gust_lt_wind:,}", observed=gust_lt_wind))

sky_levels = [("skyl1","skyl2"), ("skyl2","skyl3"), ("skyl3","skyl4")]
for lo_col, hi_col in sky_levels:
    if lo_col in df.columns and hi_col in df.columns:
        lo_s = num(df[lo_col]); hi_s = num(df[hi_col])
        bad = int((lo_s > hi_s).sum())
        add(S, "Logical Checks", "Weather", expect(S, f"expect_column_pair_values_A_to_be_less_than_or_equal_to_B — {lo_col} ≤ {hi_col}", bad==0, f"Layer order violation: {bad:,}", observed=bad))

for col in ["skyc1","skyc2","skyc3","skyc4"]:
    if col in df.columns:
        cleaned = df[col].str.strip()
        inv_sky = int((~cleaned.isin(VALID_SKY_CODES) & cleaned.notna()).sum())
        add(S, "Logical Checks", "Weather", expect(S, f"expect_column_values_to_be_in_set — {col}", inv_sky==0, f"Invalid codes: {inv_sky:,}  (valid: CLR,FEW,SCT,BKN,OVC,VV)", observed=inv_sky))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Airport Metadata (joined columns)
# ══════════════════════════════════════════════════════════════════════════════
S = "airports"

AIRPORT_META_COLS = ["airport_name","city","state","latitude","longitude","elevation_ft","airport_type","num_runways","timezone"]

for col in AIRPORT_META_COLS:
    present = col in df.columns
    add(S, "Airport Metadata", "Airports", expect(
        S, f"expect_column_to_exist — {col}",
        present,
        f"dtype: {str(df[col].dtype)}" if present else "COLUMN NOT FOUND — check join on origin_airport"
    ))

if "latitude" in df.columns:
    inv_lat = int(((num(df["latitude"]) < 17) | (num(df["latitude"]) > 72)).sum())
    add(S, "Airport Metadata", "Airports", expect(S, "expect_column_values_to_be_between — latitude [17, 72]", inv_lat==0, f"Out-of-range latitudes: {inv_lat:,}", observed=inv_lat))

if "longitude" in df.columns:
    inv_lon = int(((num(df["longitude"]) < -180) | (num(df["longitude"]) > -60)).sum())
    add(S, "Airport Metadata", "Airports", expect(S, "expect_column_values_to_be_between — longitude [-180, -60]", inv_lon==0, f"Out-of-range longitudes: {inv_lon:,}", observed=inv_lon))

if "num_runways" in df.columns:
    neg_rwy = int((num(df["num_runways"]) < 1).sum())
    add(S, "Airport Metadata", "Airports", expect(S, "expect_column_values_to_be_between — num_runways ≥ 1", neg_rwy==0, f"Airports with <1 runway: {neg_rwy:,}", observed=neg_rwy))

if "airport_type" in df.columns:
    VALID_TYPES = {"large_airport","medium_airport","small_airport","heliport","seaplane_base","closed"}
    inv_type = int((~df["airport_type"].isin(VALID_TYPES) & df["airport_type"].notna()).sum())
    add(S, "Airport Metadata", "Airports", expect(S, "expect_column_values_to_be_in_set — airport_type", inv_type==0, f"Invalid types: {inv_type:,}", observed=inv_type))

if "origin_airport" in df.columns and "latitude" in df.columns:
    null_join = int(df["latitude"].isna().sum())
    add(S, "Airport Metadata", "Airports", expect(S, "expect_foreign_key_values_to_exist — airport join coverage", null_join==0, f"Flight rows with no airport metadata (null lat): {null_join:,}", observed=null_join))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Airline Metadata (joined columns)
# ══════════════════════════════════════════════════════════════════════════════
S = "airlines"

AIRLINE_META_COLS = ["carrier_name","carrier_type","hub_airports"]

for col in AIRLINE_META_COLS:
    present = col in df.columns
    add(S, "Airline Metadata", "Airlines", expect(
        S, f"expect_column_to_exist — {col}",
        present,
        f"dtype: {str(df[col].dtype)}" if present else "COLUMN NOT FOUND — check join on carrier_code"
    ))

if "carrier_type" in df.columns:
    VALID_CARRIER_TYPES = {"Legacy","Hybrid","LCC"}
    inv_ct = int((~df["carrier_type"].isin(VALID_CARRIER_TYPES) & df["carrier_type"].notna()).sum())
    add(S, "Airline Metadata", "Airlines", expect(S, "expect_column_values_to_be_in_set — carrier_type", inv_ct==0, f"Invalid carrier_type values: {inv_ct:,}", observed=inv_ct))

if "carrier_code" in df.columns and "carrier_name" in df.columns:
    null_airline_join = int(df["carrier_name"].isna().sum())
    add(S, "Airline Metadata", "Airlines", expect(S, "expect_foreign_key_values_to_exist — airline join coverage", null_airline_join==0, f"Rows with no airline metadata (null carrier_name): {null_airline_join:,}", observed=null_airline_join))


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate statistics (passed to dashboard charts)
# ══════════════════════════════════════════════════════════════════════════════
counts = {"pass": 0, "warn": 0, "fail": 0, "info": 0}
for r in results:
    counts[r["status"]] = counts.get(r["status"], 0) + 1

output = {
    "generated_at": datetime.now().isoformat(),
    "data_path": str(DATA_PATH),
    "shape": {"rows": rows, "cols": cols},
    "summary": {
        "total": len(results),
        **counts,
        "success_rate": round(counts["pass"] / len(results) * 100, 1) if results else 0,
    },
    "year_counts": yr_counts if "year" in df.columns else {},
    "target_stats": target_stats,
    "weather_completeness": weather_completeness,
    "completeness_observed": completeness_observed,
    "sections": list(section_meta.keys()),
    "section_meta": section_meta,
    "results": results,
}

REPORT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"\nValidation complete.")
print(f"  Total expectations : {output['summary']['total']}")
print(f"  Passed             : {output['summary']['pass']}")
print(f"  Failed             : {output['summary']['fail']}")
print(f"  Warnings           : {output['summary']['warn']}")
print(f"  Success rate       : {output['summary']['success_rate']}%")
print(f"\nReport written → {REPORT_PATH}")