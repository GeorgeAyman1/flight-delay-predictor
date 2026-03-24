import pandas as pd
import os
import glob

# Paths
raw_path = "data/raw/weather"
output_path = "data/interim/weather"

# Create output folder if not exists
os.makedirs(output_path, exist_ok=True)

# Get all CSV files
files = glob.glob(os.path.join(raw_path, "*.csv"))

dataframes = []

for file in files:
    filename = os.path.basename(file)
    parts = filename.split("_")

    # Skip invalid files (like airport_timezone_mapping.csv)
    if len(parts) < 2 or not parts[1].isdigit():
        print(f"⏭️ Skipping file: {filename}")
        continue

    df = pd.read_csv(file)

    airport = parts[0]
    year = int(parts[1])

    df["airport"] = airport
    df["year"] = year

    dataframes.append(df)
# Combine all files
combined_df = pd.concat(dataframes, ignore_index=True)

# Save result
combined_df.to_csv(f"{output_path}/weather_combined.csv", index=False)

print("✅ Weather data combined successfully!")