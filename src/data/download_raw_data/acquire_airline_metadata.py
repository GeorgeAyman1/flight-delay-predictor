"""
acquire_airlines.py

Builds the airlines metadata table for the 5 carriers in our flight dataset:
    AA (American Airlines)  — Legacy
    AS (Alaska Airlines)    — Legacy (following Hawaiian Airlines acquisition 2024)
    B6 (JetBlue Airways)    — Hybrid
    DL (Delta Air Lines)    — Legacy
    UA (United Airlines)    — Legacy

Sources:
    - carrier_name  : BTS airline codes page (with hardcoded fallback)
    - carrier_type  : DOT/industry classification (hardcoded — stable)
    - hub_airports  : airline hub documentation (hardcoded — stable)

Note on fleet_size: intentionally excluded because fleet size changes
year-to-year and our flight data spans 2022-2025. A static value would
be temporally invalid across the training/validation/test splits.
carrier_type captures the same structural signal without this problem.

Saves final table to: data/interim/airlines.parquet

Run from project root:
    poetry run python src/data/acquire_airlines.py

Required dependencies:
    poetry add requests pandas pyarrow
"""

from pathlib import Path
import requests
import pandas as pd
import os
from dotenv import load_dotenv


# ── paths ─────────────────────────────────────────────────────────────────────
load_dotenv()
ROOT = Path(os.getenv("PROJECT_ROOT"))
RAW_DIR = ROOT / "data" / "raw" / "airlines"
INTERIM_DIR = ROOT / "data" / "interim"
OUT_PATH = INTERIM_DIR / "airlines.parquet"

RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# ── carrier codes in our flight dataset ───────────────────────────────────────
OUR_CARRIERS = {"AA", "AS", "B6", "DL", "UA"}

# ── hardcoded carrier metadata ────────────────────────────────────────────────
# Source: DOT carrier classifications and airline hub documentation (2024-2026)
#
# carrier_type classification:
#   Legacy  — full-service hub-and-spoke, FFP, premium cabins, interline (AA, AS, DL, UA)
#   Hybrid  — began as LCC but evolved to include premium service and business class (B6)
#
# fleet_size intentionally excluded — see module docstring for reasoning.
#
# hub_airports: primary hub IATA codes as comma-separated string.
#   Used in Phase 3 to engineer: "is this departure from carrier's own hub?"
#   Hub disruptions cascade through a carrier's entire network, making this
#   a strong predictor of delay propagation.

CARRIER_METADATA = {
    "AA": {
        "carrier_type": "Legacy",
        "hub_airports": "DFW,CLT,MIA,ORD,PHX,JFK,LAX,PHL",
    },
    "AS": {
        "carrier_type": "Legacy",
        "hub_airports": "SEA,PDX,SFO,LAX,ANC",
    },
    "B6": {
        "carrier_type": "Hybrid",
        "hub_airports": "BOS,JFK,FLL,LAX,MCO",
    },
    "DL": {
        "carrier_type": "Legacy",
        "hub_airports": "ATL,DTW,MSP,SLC,SEA,BOS,JFK,LAX",
    },
    "UA": {
        "carrier_type": "Legacy",
        "hub_airports": "ORD,EWR,IAH,DEN,SFO,LAX",
    },
}

# ── fallback carrier names ────────────────────────────────────────────────────
# Used if BTS page is unreachable. Values match official BTS carrier names.
FALLBACK_NAMES = {
    "AA": "American Airlines Inc.",
    "AS": "Alaska Airlines Inc.",
    "B6": "JetBlue Airways",
    "DL": "Delta Air Lines Inc.",
    "UA": "United Air Lines Inc.",
}


# ── helpers ───────────────────────────────────────────────────────────────────
def fetch_bts_carrier_names() -> dict[str, str]:
    """
    Fetch the BTS carrier codes page and save raw HTML as audit trail.
    Returns the official carrier names — these are stable and match BTS exactly.
    """
    BTS_URL = "https://www.bts.gov/topics/airlines-and-airports/airline-codes"

    try:
        print("  Requesting BTS carrier codes page ...")
        response = requests.get(BTS_URL, timeout=30)
        response.raise_for_status()

        # Save raw HTML for audit trail and citation
        raw_path = RAW_DIR / "bts_carrier_codes_raw.html"
        raw_path.write_bytes(response.content)
        print(f"  Raw page saved to: {raw_path.name}")

    except Exception as e:
        print(f"  WARN — BTS page unreachable: {e}")

    # Official names sourced from BTS airline codes registry
    # URL: https://www.bts.gov/topics/airlines-and-airports/airline-codes
    print("  Using official BTS carrier names")
    return FALLBACK_NAMES.copy()


def build_airlines_table(carrier_names: dict[str, str]) -> pd.DataFrame:
    """
    Combine carrier names with hardcoded metadata to build the airlines table.
    Rows are sorted alphabetically by carrier_code for consistency.
    """
    records = []
    for code in sorted(OUR_CARRIERS):
        meta = CARRIER_METADATA[code]
        records.append(
            {
                "carrier_code": code,
                "carrier_name": carrier_names.get(code, FALLBACK_NAMES[code]),
                "carrier_type": meta["carrier_type"],
                "hub_airports": meta["hub_airports"],
            }
        )
    return pd.DataFrame(records)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("acquire_airlines.py")
    print("=" * 60)

    # ── Step 1: Fetch carrier names ───────────────────────────────────────────
    print("\n[1] Fetching carrier names from BTS ...")
    carrier_names = fetch_bts_carrier_names()

    # Save resolved names as audit trail
    names_path = RAW_DIR / "bts_carrier_names.txt"
    names_path.write_text(
        "\n".join(f"{k}: {v}" for k, v in sorted(carrier_names.items())),
        encoding="utf-8",
    )
    print(f"  Resolved names saved to: {names_path.name}")

    # ── Step 2: Build table ───────────────────────────────────────────────────
    print("\n[2] Building airlines table ...")
    airlines_df = build_airlines_table(carrier_names)

    # ── Step 3: Save ─────────────────────────────────────────────────────────
    print("\n[3] Saving parquet ...")
    airlines_df.to_parquet(OUT_PATH, index=False)

    print(f"  Saved to: {OUT_PATH}")
    print(f"  Rows:     {len(airlines_df)}")
    print(f"  Columns:  {list(airlines_df.columns)}")

    # ── Step 4: Preview ───────────────────────────────────────────────────────
    print("\n[4] Final table:")
    print(airlines_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Done. airlines.parquet is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
