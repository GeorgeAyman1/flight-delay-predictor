# Cleaning Summary

**Input file:** `C:\Users\VICTUS\Desktop\Engineering\Sem 8\Data Science\flight-delay-predictor\data\interim\merged_dataset.parquet`
**Output file:** `C:\Users\VICTUS\Desktop\Engineering\Sem 8\Data Science\flight-delay-predictor\data\processed\cleaned_merged_dataset_v2.parquet`

## Shape

- Before: **4,977,182 rows × 60 columns**
- After: **4,977,182 rows × 61 columns**
- Rows removed: **0**

## Logged Cleaning Steps

| Step | Action | Columns | Rows Affected | Details | Reason |
|---:|---|---|---:|---|---|
| 1 | Standardization | all object/string columns | 26931 | Trimmed leading/trailing whitespace from text columns. | Whitespace can break joins, filters, grouping, and categorical consistency. |
| 2 | Standardization | carrier_code, origin_airport, destination_airport | 0 | Uppercased carrier and airport codes. | Codes should be case-consistent before validation and modeling. |
| 3 | Type coercion | date_dt, scheduled_departure_dt, valid | 0 | Converted confirmed date/datetime columns to pandas datetime. | Only true timestamps were parsed. HHMM operational time fields and timezone were left untouched. |
| 4 | Preservation | scheduled_departure_time, actual_departure_time, wheelsoff_time, timezone | 0 | Kept operational HHMM-style time fields and timezone as strings instead of parsing them as datetimes. | These are identifiers/clock fields, not full timestamps. |
| 5 | Type coercion | departure_delay_minutes, tmpf, dwpf, sknt, gust, relh, vsby, p01i, alti, flight_number, scheduled_elapsed_time_minutes, actual_elapsed_time_minutes, taxiout_time_minutes, delay_carrier_minutes, delay_weather_minutes, delay_national_aviation_system_minutes, delay_security_minutes, delay_late_aircraft_arrival_minutes, year_x, lon, lat, elevation, year_y, latitude, longitude, elevation_ft, num_runways | 0 | Converted likely numeric columns to numeric dtype using coercion. | Numeric consistency is required for anomaly checks, summaries, and modeling. |
| 6 | Removal | all columns | 0 | No exact duplicate rows found. | Exact duplicates add no information and can bias analysis. |
| 7 | Audit | date_dt, carrier_code, flight_number, origin_airport, destination_airport, scheduled_departure_dt | 0 | Checked business-key duplicates only; no rows dropped because DROP_FLIGHT_KEY_DUPLICATES=False. | Business-key duplicate removal should be explicit, not accidental. |
| 8 | Removal | carrier_code | 0 | Removed the known BTS source/junk row and carrier codes outside the 5 project carriers. | Keeps the dataset aligned with the documented project scope. |
| 9 | Cell nullification | dwpf | 41 | Set `dwpf` to missing where `dwpf > tmpf`. | Dew point above air temperature indicates a bad sensor reading. |
| 10 | Cell nullification | gust | 4 | Set `gust` to missing where `gust < sknt`. | Wind gust lower than sustained wind indicates a bad sensor reading. |
| 11 | Cell nullification | relh | 0 | Set `relh` to missing where values were outside [0, 100]. | Clearly impossible physical values should not be retained as valid measurements. |
| 12 | Cell nullification | sknt | 0 | Set `sknt` to missing where values were negative. | Clearly impossible physical values should not be retained as valid measurements. |
| 13 | Cell nullification | gust | 0 | Set `gust` to missing where values were negative. | Clearly impossible physical values should not be retained as valid measurements. |
| 14 | Cell nullification | vsby | 0 | Set `vsby` to missing where values were negative. | Clearly impossible physical values should not be retained as valid measurements. |
| 15 | Cell nullification | p01i | 0 | Set `p01i` to missing where values were negative. | Clearly impossible physical values should not be retained as valid measurements. |
| 16 | Feature creation | departure_delayed | 4977182 | Created the binary target from `departure_delay_minutes` using threshold >= 15 minutes. | This is the project target used for classification. |
| 17 | Documentation | gust, wxcodes, skyc2, skyc3, skyc4, skyl2, skyl3, skyl4 | 30254561 | Preserved natural sparsity in sparse weather fields; no blanket imputation was performed here. | These columns are sparse by nature, so missingness is meaningful. |
| 18 | Dataset split | year-based temporal split | 4977182 | Split cleaned dataset into train (2022-2023), validation (2024), and test (2025). | Prevents temporal leakage and reflects real-world forecasting conditions. |

## Missing Values (Top 20 Before)

```
skyl4          4759421
skyc4          4759421
wxcodes        4422737
gust           4224096
skyc3          3767285
skyl3          3767285
skyl2          2277156
skyc2          2277156
skyl1           633078
p01i            259195
drct            175587
mslp             21328
tail_number      19062
sknt              5042
feel              1058
relh               534
dwpf               493
tmpf               433
vsby               240
alti               142
```

## Missing Values (Top 20 After)

```
skyc4          4759421
skyl4          4759421
wxcodes        4422737
gust           4224100
skyl3          3767285
skyc3          3767285
skyc2          2277156
skyl2          2277156
skyl1           633078
p01i            259195
drct            175587
mslp             21328
tail_number      19062
sknt              5042
feel              1058
dwpf               534
relh               534
tmpf               433
vsby               240
alti               142
```

## Sparse Weather Fields Left As-Is

These were not treated as ordinary errors because their missingness can be meaningful.

```
gust       4224100
wxcodes    4422737
skyc2      2277156
skyc3      3767285
skyc4      4759421
skyl2      2277156
skyl3      3767285
skyl4      4759421
```

## Data Types Before

```
carrier_code                                         str
date_mmddyyyy                             datetime64[us]
flight_number                                    float64
tail_number                                          str
destination_airport                                  str
scheduled_departure_time                             str
actual_departure_time                                str
scheduled_elapsed_time_minutes                   float64
actual_elapsed_time_minutes                      float64
departure_delay_minutes                          float64
wheelsoff_time                                       str
taxiout_time_minutes                             float64
delay_carrier_minutes                            float64
delay_weather_minutes                            float64
delay_national_aviation_system_minutes           float64
delay_security_minutes                           float64
delay_late_aircraft_arrival_minutes              float64
origin_airport                                       str
airline                                              str
year_x                                           float64
date_dt                                   datetime64[us]
scheduled_departure_dt                    datetime64[us]
station                                              str
valid                                     datetime64[us]
lon                                              float64
lat                                              float64
elevation                                        float64
tmpf                                             float64
dwpf                                             float64
relh                                             float64
drct                                             float64
sknt                                             float64
gust                                             float64
p01i                                             float64
alti                                             float64
mslp                                             float64
vsby                                             float64
feel                                             float64
wxcodes                                              str
skyc1                                                str
skyc2                                                str
skyc3                                                str
skyc4                                                str
skyl1                                            float64
skyl2                                            float64
skyl3                                            float64
skyl4                                            float64
year_y                                           float64
carrier_name                                         str
carrier_type                                         str
hub_airports                                         str
airport_name                                         str
city                                                 str
state                                                str
latitude                                         float64
longitude                                        float64
elevation_ft                                     float64
airport_type                                         str
num_runways                                        int64
timezone                                             str
```

## Data Types After

```
carrier_code                                      string
date_mmddyyyy                             datetime64[us]
flight_number                                    Float64
tail_number                                       string
destination_airport                               string
scheduled_departure_time                          string
actual_departure_time                             string
scheduled_elapsed_time_minutes                   Float64
actual_elapsed_time_minutes                      Float64
departure_delay_minutes                          Float64
wheelsoff_time                                    string
taxiout_time_minutes                             Float64
delay_carrier_minutes                            Float64
delay_weather_minutes                            Float64
delay_national_aviation_system_minutes           Float64
delay_security_minutes                           Float64
delay_late_aircraft_arrival_minutes              Float64
origin_airport                                    string
airline                                           string
year_x                                           Float64
date_dt                                   datetime64[us]
scheduled_departure_dt                    datetime64[us]
station                                           string
valid                                     datetime64[us]
lon                                              Float64
lat                                              Float64
elevation                                        Float64
tmpf                                             Float64
dwpf                                             Float64
relh                                             Float64
drct                                             float64
sknt                                             Float64
gust                                             Float64
p01i                                             Float64
alti                                             Float64
mslp                                             float64
vsby                                             Float64
feel                                             float64
wxcodes                                           string
skyc1                                             string
skyc2                                             string
skyc3                                             string
skyc4                                             string
skyl1                                            float64
skyl2                                            float64
skyl3                                            float64
skyl4                                            float64
year_y                                           Float64
carrier_name                                      string
carrier_type                                      string
hub_airports                                      string
airport_name                                      string
city                                              string
state                                             string
latitude                                         Float64
longitude                                        Float64
elevation_ft                                     Float64
airport_type                                      string
num_runways                                        Int64
timezone                                          string
departure_delayed                                   Int8
```

## Temporal Split Summary

```
train_rows         2454517
validation_rows    1314163
test_rows          1208502
```

## Target Distribution

```
on_time    3824256
delayed    1152926
```

## Split Target Distribution

```
{
  "train": {
    "0": 1896894,
    "1": 557623
  },
  "validation": {
    "0": 1000199,
    "1": 313964
  },
  "test": {
    "0": 927163,
    "1": 281339
  }
}
```