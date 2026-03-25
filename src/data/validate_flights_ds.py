from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/interim/flights_combined.parquet")
df = pd.read_parquet(DATA_PATH)

# check_shape() check_duplicates() check_year_distribution() check_target_distribution()

# Rows and Columns
rows = len(df)
cols = len(df.columns)
print("Number of rows=", rows)
print("Number of columns=", cols)

# Schema
print(df.columns.tolist())
print(df.dtypes)

# Number of missing data of important fields ----> Completeness
columns = [
    "carrier_code",
    "origin_airport",
    "destination_airport",
    "date_mmddyyyy",
    "scheduled_departure_time",
    "scheduled_elapsed_time_minutes",
    "airline",
    "year",
    "flight_number",
]
missingData = df[columns].isnull().any(axis=1).sum()
print("Number of rows with missing important data --> ", missingData, " divided into:")
missingcC = df["carrier_code"].isnull().sum()
print("Carrier code=", missingcC)
missingoA = df["origin_airport"].isnull().sum()
print("Origin Airport=", missingoA)
missingdA = df["destination_airport"].isnull().sum()
print("Destination Airport=", missingdA)
missingD = df["date_mmddyyyy"].isnull().sum()
print("Date=", missingD)
missingsD = df["scheduled_departure_time"].isnull().sum()
print("Scheduled Departure=", missingsD)
missingsE = df["scheduled_elapsed_time_minutes"].isnull().sum()
print("Scheduled elapsed=", missingsE)
missingA = df["airline"].isnull().sum()
print("Airline=", missingA)
missingY = df["year"].isnull().sum()
print("Year=", missingY)
missingfN = df["flight_number"].isnull().sum()
print("Flight Number=", missingfN)

# Year Distributions: 22-23-24-25-NaN
yearsNaN = df[df["year"].isna()]
count = len(yearsNaN)
print("Years of NaN=", count)

years2022 = df[df["year"] == 2022]
print("Years of 2022=", len(years2022))

years2023 = df[df["year"] == 2023]
print("Years of 2023=", len(years2023))

years2024 = df[df["year"] == 2024]
print("Years of 2024=", len(years2024))

years2025 = df[df["year"] == 2025]
print("Years of 2025=", len(years2025))

invYears = df[df["year"].notna() & ((df["year"] > 2025) | (df["year"] < 2022))]
print("Years outside of range =", len(invYears))

# Number of duplicates
duplicates = df.duplicated().sum()
print("Exact duplicate rows =", duplicates)

# Number of potential duplicate flight records
flight_id_cols = [
    "date_mmddyyyy",
    "carrier_code",
    "flight_number",
    "origin_airport",
    "destination_airport",
    "scheduled_departure_time",
]

subset_duplicates = df.duplicated(subset=flight_id_cols).sum()
print("Duplicate rows based on flight identity columns =", subset_duplicates)

# Some validity checks
negsE = df[df["scheduled_elapsed_time_minutes"] <= 0]
print("Negative scheduled elapsed=", len(negsE))

times = pd.to_numeric(df["scheduled_departure_time"], errors="coerce")
invalid_time = (
    times.notna() & ((times < 0) | (times > 2359) | ((times % 100) >= 60))
).sum()
print("Invalid scheduled departure time=", invalid_time)

invalid_origin = (
    ~df["origin_airport"].astype(str).str.fullmatch(r"[A-Z]{3}")
    & df["origin_airport"].notna()
).sum()
print("Invalid origin airport codes =", invalid_origin)

invalid_dest = (
    ~df["destination_airport"].astype(str).str.fullmatch(r"[A-Z]{3}")
    & df["destination_airport"].notna()
).sum()
print("Invalid destination airport codes =", invalid_dest)

invalid_carrier = (
    ~df["carrier_code"].astype(str).str.fullmatch(r"[A-Z0-9]{2}")
    & df["carrier_code"].notna()
).sum()
print("Invalid carrier codes =", invalid_carrier)

# Target validation
tar = df[df["departure_delay_minutes"].isna()]
print("Number of missing target values=", len(tar))

print("Target data type =", df["departure_delay_minutes"].dtype)

negrecs = df[df["departure_delay_minutes"] < 0]
print("Number of negative targets=", len(negrecs))

posrecs = df[df["departure_delay_minutes"] >= 0]
print("Number of non-negative targets=", len(posrecs))

print("\nBasic statistics for departure_delay_minutes:")
print(df["departure_delay_minutes"].describe())

del15 = df[df["departure_delay_minutes"] >= 15]
print("Number of targets greater than 15=", len(del15))

notdel15 = df[df["departure_delay_minutes"] < 15]
print("Number of targets less than 15=", len(notdel15))
