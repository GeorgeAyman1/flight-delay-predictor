from pathlib import Path
import re
import pandas as pd

RAW_DIR = Path("data/raw/flights")
OUT_PATH = Path("data/interim/flights_combined.parquet")


def normalize_column_name(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^\w\s]", "", col)
    col = re.sub(r"\s+", "_", col)
    return col


def parse_filename(file_path: Path) -> tuple[str, str]:
    # Example: BOS_B6_raw.csv
    parts = re.split(r"[_\-\s]+", file_path.stem)

    if len(parts) < 2:
        raise ValueError(f"Bad filename format: {file_path.name}")

    airport = parts[0].upper()
    airline = parts[1].upper()
    return airport, airline


def find_header_row(file_path: Path) -> int:
    """
    Find the line index where the real CSV header starts.
    We look for a line containing known column names.
    """
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            if "Carrier Code" in line and "Date (MM/DD/YYYY)" in line:
                return i

    raise ValueError(f"Could not find header row in {file_path.name}")


def read_raw_file(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV files are allowed. Got: {file_path.name}")

    header_row = find_header_row(file_path)

    df = pd.read_csv(
        file_path,
        skiprows=header_row,
        encoding="utf-8-sig"
    )

    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


def main():
    files = sorted(list(RAW_DIR.glob("*.csv")))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    print("Files found:")
    for f in files:
        print(f" - {f.name}")

    all_dfs = []

    for file_path in files:
        airport, airline = parse_filename(file_path)
        df = read_raw_file(file_path)

        if "date_mmddyyyy" not in df.columns:
            raise KeyError(
                f"'Date (MM/DD/YYYY)' not found in {file_path.name}. "
                f"Columns found: {list(df.columns)}"
            )

        df["date_mmddyyyy"] = pd.to_datetime(
            df["date_mmddyyyy"],
            format="%m/%d/%Y",
            errors="coerce"
        )

        df["origin_airport"] = airport
        df["airline"] = airline
        df["year"] = df["date_mmddyyyy"].dt.year

        all_dfs.append(df)
        print(f"Loaded {file_path.name}: {len(df):,} rows")

    combined = pd.concat(all_dfs, ignore_index=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)

    print("\nDone.")
    print(f"Saved to: {OUT_PATH}")
    print(f"Rows: {len(combined):,}")
    print(f"Columns: {combined.shape[1]}")
    print("\nYears:")
    print(combined["year"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()