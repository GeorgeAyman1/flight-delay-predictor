from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/interim/weather/weather_combined.csv")

df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.lower()

# shape
rows = len(df)
cols = len(df.columns)
print("Number of rows=", rows)
print("Number of columns=", cols)

# schema
print(df.columns.tolist())
print(df.dtypes)

print("____________________________________________________________________")

# Completeness (critical fields)
critical_cols = ["station", "valid", "tmpf", "dwpf", "relh", "sknt", "vsby"]
missingData = df[critical_cols].isnull().any(axis=1).sum()
print("Number of rows with missing critical data --> ", missingData, " divided into:")

missingStation = df["station"].isnull().sum()
print("Station=", missingStation)
missingValid = df["valid"].isnull().sum()
print("Valid (timestamp)=", missingValid)
missingTmpf = df["tmpf"].isnull().sum()
print("Temperature (tmpf)=", missingTmpf)
missingDwpf = df["dwpf"].isnull().sum()
print("Dew Point (dwpf)=", missingDwpf)
missingRelh = df["relh"].isnull().sum()
print("Relative Humidity (relh)=", missingRelh)
missingSknt = df["sknt"].isnull().sum()
print("Wind Speed (sknt)=", missingSknt)
missingVsby = df["vsby"].isnull().sum()
print("Visibility (vsby)=", missingVsby)


print("____________________________________________________________________")

# Missing values across ALL weather fields
WEATHER_FIELDS = [
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "gust",
    "p01i",
    "alti",
    "mslp",
    "vsby",
    "feel",
    "wxcodes",
    "skyc1",
    "skyc2",
    "skyc3",
    "skyc4",
    "skyl1",
    "skyl2",
    "skyl3",
    "skyl4",
]

print("Missing values per weather field:")
for col in WEATHER_FIELDS:
    if col in df.columns:
        count = df[col].isnull().sum()
        pct = count / rows * 100
        print(f"  {col} missing= {count}  ({pct:.1f}%)")
    else:
        print(f"  {col} = COLUMN NOT FOUND")

print("____________________________________________________________________")

# Duplicates
duplicates = df.duplicated().sum()
print("Exact duplicate rows =", duplicates)

obs_id_cols = ["station", "valid"]
subset_duplicates = df.duplicated(subset=obs_id_cols).sum()
print("Duplicate rows based on (station, valid timestamp) =", subset_duplicates)

print("____________________________________________________________________")

# Year distribution
df["_year"] = pd.to_datetime(df["valid"], errors="coerce").dt.year

yearsNaN = df[df["_year"].isna()]
print("Years of NaN=", len(yearsNaN))

years2022 = df[df["_year"] == 2022]
print("Years of 2022=", len(years2022))

years2023 = df[df["_year"] == 2023]
print("Years of 2023=", len(years2023))

years2024 = df[df["_year"] == 2024]
print("Years of 2024=", len(years2024))

years2025 = df[df["_year"] == 2025]
print("Years of 2025=", len(years2025))

invYears = df[df["_year"].notna() & ((df["_year"] < 2022) | (df["_year"] > 2025))]
print("Years outside of range =", len(invYears))


print("____________________________________________________________________")

# Value range checks
negTmpf = df[pd.to_numeric(df["tmpf"], errors="coerce") < -80]
print("Temperature below -80F =", len(negTmpf))

invalidRelh = df[
    pd.to_numeric(df["relh"], errors="coerce").pipe(lambda s: (s < 0) | (s > 100))
]
print("Humidity outside [0, 100] =", len(invalidRelh))

invalidDrct = df[
    pd.to_numeric(df["drct"], errors="coerce").pipe(lambda s: (s < 0) | (s > 360))
]
print("Wind direction outside [0, 360] =", len(invalidDrct))

negSknt = df[pd.to_numeric(df["sknt"], errors="coerce") < 0]
print("Negative wind speed =", len(negSknt))

negGust = df[pd.to_numeric(df["gust"], errors="coerce") < 0]
print("Negative wind gust =", len(negGust))

negP01i = df[pd.to_numeric(df["p01i"], errors="coerce") < 0]
print("Negative precipitation =", len(negP01i))

invalidAlti = df[
    pd.to_numeric(df["alti"], errors="coerce").pipe(lambda s: (s < 25) | (s > 32))
]
print("Altimeter outside [25, 32] inHg =", len(invalidAlti))

invalidMslp = df[
    pd.to_numeric(df["mslp"], errors="coerce").pipe(lambda s: (s < 850) | (s > 1100))
]
print("Sea level pressure outside [850, 1100] mb =", len(invalidMslp))

invalidVsby = df[
    pd.to_numeric(df["vsby"], errors="coerce").pipe(lambda s: (s < 0) | (s > 100))
]
print("Visibility outside [0, 100] miles =", len(invalidVsby))


print("____________________________________________________________________")


# Categorical checks (sky cover codes)
VALID_SKY_CODES = {"CLR", "FEW", "SCT", "BKN", "OVC", "VV"}

skyc1_invalid = (~df["skyc1"].isin(VALID_SKY_CODES) & df["skyc1"].notna()).sum()
print("Invalid skyc1 codes =", skyc1_invalid)

skyc2_invalid = (~df["skyc2"].isin(VALID_SKY_CODES) & df["skyc2"].notna()).sum()
print("Invalid skyc2 codes =", skyc2_invalid)

skyc3_invalid = (~df["skyc3"].isin(VALID_SKY_CODES) & df["skyc3"].notna()).sum()
print("Invalid skyc3 codes =", skyc3_invalid)

skyc4_invalid = (~df["skyc4"].isin(VALID_SKY_CODES) & df["skyc4"].notna()).sum()
print("Invalid skyc4 codes =", skyc4_invalid)

print("____________________________________________________________________")

# Logical consistency

# Dew point should never be higher than temperature
tmpf = pd.to_numeric(df["tmpf"], errors="coerce")
dwpf = pd.to_numeric(df["dwpf"], errors="coerce")
dewGtTemp = ((dwpf - tmpf) > 0).sum()
print("Rows where dew point > temperature (impossible)=", dewGtTemp)

# Wind gust should always be >= wind speed
sknt = pd.to_numeric(df["sknt"], errors="coerce")
gust = pd.to_numeric(df["gust"], errors="coerce")
gustLtWind = (gust < sknt).sum()
print("Rows where gust < wind speed (impossible)=", gustLtWind)

# Sky layer heights should go from low to high (skyl1 < skyl2 < skyl3 < skyl4)
skyl1 = pd.to_numeric(df["skyl1"], errors="coerce")
skyl2 = pd.to_numeric(df["skyl2"], errors="coerce")
skyl3 = pd.to_numeric(df["skyl3"], errors="coerce")
skyl4 = pd.to_numeric(df["skyl4"], errors="coerce")

badOrder12 = (skyl1 > skyl2).sum()
print("Rows where skyl1 > skyl2=", badOrder12)

badOrder23 = (skyl2 > skyl3).sum()
print("Rows where skyl2 > skyl3=", badOrder23)

badOrder34 = (skyl3 > skyl4).sum()
print("Rows where skyl3 > skyl4=", badOrder34)

print("____________________________________________________________________")

# Key field statistics
print("\nBasic statistics for tmpf:")
print(df["tmpf"].describe())

print("\nBasic statistics for relh:")
print(df["relh"].describe())

print("\nBasic statistics for sknt:")
print(df["sknt"].describe())

print("\nBasic statistics for vsby:")
print(df["vsby"].describe())

print("\nBasic statistics for p01i:")
print(df["p01i"].describe())
