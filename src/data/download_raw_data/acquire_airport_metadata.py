"""
acquire_airports.py

Builds the airports metadata table from two sources:
  1. OurAirports airports.csv  — code, name, city, state, lat, lon, elevation, type
  2. OurAirports runways.csv   — runway count per airport
  3. timezonefinder (library)  — IANA timezone derived from coordinates

Saves final table to: data/interim/airports.parquet

Run from project root:
    poetry run python src/data/acquire_airports.py

Required dependencies:
    poetry add requests pandas pyarrow timezonefinder
"""

from pathlib import Path
import requests
import pandas as pd
from timezonefinder import TimezoneFinder
import os
from dotenv import load_dotenv

load_dotenv()

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(os.getenv("PROJECT_ROOT"))
RAW_DIR = ROOT / "data" / "raw" / "airports"
INTERIM_DIR = ROOT / "data" / "interim"
OUT_PATH = INTERIM_DIR / "airports.parquet"

RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# ── source URLs ───────────────────────────────────────────────────────────────
OURAIRPORTS_AIRPORTS_URL = (
    "https://davidmegginson.github.io/ourairports-data/airports.csv"
)
OURAIRPORTS_RUNWAYS_URL = (
    "https://davidmegginson.github.io/ourairports-data/runways.csv"
)

# ── all airport codes in our flight dataset ───────────────────────────────────
# Union of all unique origin + destination airport codes from flights_combined.
# Update this set if the flight dataset changes.
OUR_AIRPORTS = {
    "ABQ",
    "ACK",
    "AGS",
    "ALB",
    "AMA",
    "ANC",
    "ATL",
    "ATW",
    "AUS",
    "AVL",
    "AVP",
    "BDL",
    "BFL",
    "BGR",
    "BHM",
    "BIL",
    "BIS",
    "BNA",
    "BOI",
    "BOS",
    "BQN",
    "BTR",
    "BTV",
    "BUF",
    "BUR",
    "BWI",
    "BZN",
    "CAE",
    "CHA",
    "CHS",
    "CID",
    "CLE",
    "CLT",
    "CMH",
    "COS",
    "CVG",
    "DAB",
    "DAL",
    "DAY",
    "DCA",
    "DEN",
    "DFW",
    "DLH",
    "DRO",
    "DSM",
    "DTW",
    "ECP",
    "EGE",
    "ELP",
    "EUG",
    "EWR",
    "EYW",
    "FAI",
    "FAR",
    "FAT",
    "FCA",
    "FLL",
    "FSD",
    "GEG",
    "GFK",
    "GJT",
    "GRB",
    "GRR",
    "GSO",
    "GSP",
    "GTF",
    "GUC",
    "HDN",
    "HNL",
    "HOU",
    "HRL",
    "HSV",
    "HYA",
    "IAD",
    "IAH",
    "ICT",
    "ILM",
    "IND",
    "ISP",
    "JAC",
    "JAX",
    "JFK",
    "JNU",
    "KOA",
    "KTN",
    "LAS",
    "LAX",
    "LBB",
    "LEX",
    "LGA",
    "LIH",
    "LIT",
    "LNK",
    "MAF",
    "MCI",
    "MCO",
    "MDT",
    "MDW",
    "MEM",
    "MFE",
    "MFR",
    "MHT",
    "MIA",
    "MKE",
    "MRY",
    "MSN",
    "MSO",
    "MSP",
    "MSY",
    "MTJ",
    "MVY",
    "MYR",
    "OAK",
    "OGG",
    "OKC",
    "OMA",
    "ONT",
    "ORD",
    "ORF",
    "ORH",
    "PAE",
    "PBI",
    "PDX",
    "PHL",
    "PHX",
    "PIT",
    "PNS",
    "PQI",
    "PSC",
    "PSE",
    "PSP",
    "PVD",
    "PWM",
    "RAP",
    "RDM",
    "RDU",
    "RIC",
    "RNO",
    "ROC",
    "RST",
    "RSW",
    "SAN",
    "SAT",
    "SAV",
    "SBA",
    "SBN",
    "SBP",
    "SDF",
    "SEA",
    "SFO",
    "SIT",
    "SJC",
    "SJU",
    "SLC",
    "SMF",
    "SNA",
    "SRQ",
    "STL",
    "STS",
    "STT",
    "STX",
    "SYR",
    "TLH",
    "TPA",
    "TUL",
    "TUS",
    "TVC",
    "TYS",
    "VPS",
    "WRG",
    "XNA",
}


# ── helpers ───────────────────────────────────────────────────────────────────
def download_file(url: str, dest: Path, label: str) -> None:
    """Download url to dest. Skips silently if file already exists."""
    if dest.exists():
        print(f"  [skip] {label} already downloaded: {dest.name}")
        return
    print(f"  Downloading {label} ...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"  Saved: {dest.name}  ({len(response.content):,} bytes)")


def load_airports(path: Path) -> pd.DataFrame:
    """
    Load OurAirports airports.csv.
    Filters to airports in OUR_AIRPORTS and selects + renames needed columns.
    """
    df = pd.read_csv(path, low_memory=False)

    # Filter to only airports in our dataset
    df = df[df["iata_code"].isin(OUR_AIRPORTS)].copy()

    # Extract 2-letter state code from iso_region e.g. "US-CA" -> "CA"
    df["state"] = df["iso_region"].str.split("-").str[-1]

    df = df.rename(
        columns={
            "iata_code": "airport_code",
            "name": "airport_name",
            "municipality": "city",
            "latitude_deg": "latitude",
            "longitude_deg": "longitude",
            "elevation_ft": "elevation_ft",
            "type": "airport_type",
            "ident": "ident",
        }
    )

    return df[
        [
            "airport_code",
            "airport_name",
            "city",
            "state",
            "latitude",
            "longitude",
            "elevation_ft",
            "airport_type",
            "ident",
        ]
    ].reset_index(drop=True)


def load_runway_counts(runways_path: Path, airports_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count runways per airport from OurAirports runways.csv.
    Joins on 'ident' (ICAO-style code e.g. KJFK) since runways.csv
    does not use IATA codes.
    Returns: airport_code -> num_runways.
    """
    rwy_df = pd.read_csv(runways_path, low_memory=False)

    counts = (
        rwy_df[rwy_df["airport_ident"].isin(airports_df["ident"])]
        .groupby("airport_ident")
        .size()
        .reset_index(name="num_runways")
        .rename(columns={"airport_ident": "ident"})
    )

    result = airports_df[["airport_code", "ident"]].merge(
        counts, on="ident", how="left"
    )
    result["num_runways"] = result["num_runways"].fillna(0).astype(int)

    return result[["airport_code", "num_runways"]]


def derive_timezones(df: pd.DataFrame) -> pd.Series:
    """
    Derive IANA timezone string for each airport from its lat/lon.
    Uses the timezonefinder library — no API call, works fully offline.
    Example output: "America/New_York", "America/Los_Angeles"
    """
    tf = TimezoneFinder()
    timezones = []

    for _, row in df.iterrows():
        tz = tf.timezone_at(lat=float(row["latitude"]), lng=float(row["longitude"]))
        timezones.append(tz if tz else "Unknown")

    return pd.Series(timezones, index=df.index, name="timezone")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("acquire_airports.py")
    print("=" * 60)

    # ── Step 1: Download raw files ────────────────────────────────────────────
    print("\n[1] Downloading raw files ...")
    airports_raw = RAW_DIR / "ourairports_airports.csv"
    runways_raw = RAW_DIR / "ourairports_runways.csv"

    download_file(OURAIRPORTS_AIRPORTS_URL, airports_raw, "OurAirports airports")
    download_file(OURAIRPORTS_RUNWAYS_URL, runways_raw, "OurAirports runways")

    # ── Step 2: Load and filter airports ──────────────────────────────────────
    print("\n[2] Loading airports ...")
    airports_df = load_airports(airports_raw)

    matched = set(airports_df["airport_code"])
    not_found = OUR_AIRPORTS - matched

    print(f"  Matched:   {len(matched)} / {len(OUR_AIRPORTS)}")
    if not_found:
        print(f"  Not found in OurAirports ({len(not_found)}):")
        for code in sorted(not_found):
            print(f"    - {code}")

    # ── Step 3: Load runway counts ────────────────────────────────────────────
    print("\n[3] Loading runway counts ...")
    runway_df = load_runway_counts(runways_raw, airports_df)
    airports_df = airports_df.merge(runway_df, on="airport_code", how="left")
    airports_df["num_runways"] = airports_df["num_runways"].fillna(0).astype(int)

    no_runway = (airports_df["num_runways"] == 0).sum()
    if no_runway:
        print(f"  WARNING: {no_runway} airports have 0 runways in OurAirports data")
    print(f"  Runway counts loaded for {len(airports_df) - no_runway} airports")

    # ── Step 4: Derive timezones ──────────────────────────────────────────────
    print("\n[4] Deriving timezones from coordinates ...")
    airports_df["timezone"] = derive_timezones(airports_df)

    unknown_tz = (airports_df["timezone"] == "Unknown").sum()
    if unknown_tz:
        print(f"  WARNING: {unknown_tz} airports have unknown timezone")
    else:
        print(f"  All {len(airports_df)} airports assigned a timezone successfully")

    # ── Step 5: Handle nulls ──────────────────────────────────────────────────
    print("\n[5] Handling nulls ...")
    null_elev = airports_df["elevation_ft"].isna().sum()
    if null_elev:
        median_elev = airports_df["elevation_ft"].median()
        airports_df["elevation_ft"] = airports_df["elevation_ft"].fillna(median_elev)
        print(
            f"  elevation_ft: {null_elev} nulls filled with median ({median_elev:.0f} ft)"
        )
    else:
        print("  No nulls found in any column")

    # ── Step 6: Final column selection ───────────────────────────────────────
    final_cols = [
        "airport_code",
        "airport_name",
        "city",
        "state",
        "latitude",
        "longitude",
        "elevation_ft",
        "airport_type",
        "num_runways",
        "timezone",
    ]

    # Drop ident — only needed for runway join, not part of final schema
    airports_df = airports_df[final_cols]

    # ── Step 7: Save ──────────────────────────────────────────────────────────
    print("\n[6] Saving final parquet ...")
    airports_df.to_parquet(OUT_PATH, index=False)

    print(f"  Saved to:  {OUT_PATH}")
    print(f"  Rows:      {len(airports_df)}")
    print(f"  Columns:   {airports_df.shape[1]}")
    print(f"  Columns:   {list(airports_df.columns)}")

    print("\n[7] Sample rows:")
    print(airports_df.head(5).to_string(index=False))

    print("\n" + "=" * 60)
    print("Done. airports.parquet is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
