# Cleaning Summary

**Input file:** `C:\Users\VICTUS\Desktop\Engineering\Sem 8\Data Science\flight-delay-predictor\data\interim\merged_dataset.parquet`
**Output file:** `C:\Users\VICTUS\Desktop\Engineering\Sem 8\Data Science\flight-delay-predictor\data\processed\cleaned_merged_dataset.parquet`

## Shape

- Before: **4,977,182 rows × 60 columns**
- After: **4,977,182 rows × 61 columns**
- Rows removed: **0**

## Logged Cleaning Steps

| Step | Action | Columns | Rows Affected | Details | Reason |
|---:|---|---|---:|---|---|
| 1 | Standardization | all object/string columns | 26931 | Trimmed leading/trailing whitespace from text columns. | Whitespace can break joins, filters, grouping, and categorical consistency. |
| 2 | Standardization | carrier_code, origin_airport, destination_airport | 0 | Uppercased code-like identifiers. | Carrier and airport codes should be case-consistent before validation or modeling. |
| 3 | Type coercion | scheduled_departure_dt, valid | 0 | Converted confirmed date/datetime columns to pandas datetime. | Only true date/datetime fields should be parsed as timestamps. HHMM-style time columns are left numeric. |
| 4 | Type coercion | departure_delay_minutes, tmpf, dwpf, sknt, gust, relh, vsby, p01i, alti, scheduled_elapsed_time_minutes, actual_elapsed_time_minutes, taxiout_time_minutes, delay_carrier_minutes, delay_weather_minutes, delay_national_aviation_system_minutes, delay_security_minutes, delay_late_aircraft_arrival_minutes, elevation, latitude, longitude, elevation_ft, num_runways | 0 | Converted likely numeric columns to numeric dtype using coercion. | Numeric consistency is required for statistics, anomaly checks, and modeling. |
| 5 | Removal | all columns | 0 | No exact duplicate rows found. | Exact duplicates add no information and can bias analysis. |
| 6 | Audit | carrier_code, flight_number, origin_airport, destination_airport, scheduled_departure_dt | 0 | Checked business-key duplicates only; no rows dropped because DROP_FLIGHT_KEY_DUPLICATES=False. | Business-key duplicate removal should be explicit, not accidental. |
| 7 | Removal | carrier_code | 0 | Removed known BTS source/junk row and any carrier codes outside the 5 project carriers. | Keeps the dataset aligned with the documented project scope. |
| 8 | Cell nullification | dwpf | 41 | Set `dwpf` to missing where `dwpf > tmpf`. | Dew point above air temperature indicates a bad weather reading. |
| 9 | Cell nullification | gust | 4 | Set `gust` to missing where `gust < sknt`. | Wind gust lower than sustained wind indicates a bad weather reading. |
| 10 | Cell nullification | relh | 0 | Set `relh` to missing where values were outside [0, 100]. | Clearly impossible physical values should not be kept as valid measurements. |
| 11 | Cell nullification | sknt | 0 | Set `sknt` to missing where values were negative. | Clearly impossible physical values should not be kept as valid measurements. |
| 12 | Cell nullification | gust | 0 | Set `gust` to missing where values were negative. | Clearly impossible physical values should not be kept as valid measurements. |
| 13 | Cell nullification | vsby | 0 | Set `vsby` to missing where values were negative. | Clearly impossible physical values should not be kept as valid measurements. |
| 14 | Cell nullification | p01i | 0 | Set `p01i` to missing where values were negative. | Clearly impossible physical values should not be kept as valid measurements. |
| 15 | Feature creation | departure_delayed | 4977182 | Created/standardized binary target from `departure_delay_minutes` using threshold >= 15 minutes. | This is the project target used for classification. |
| 16 | Documentation | gust, wxcodes, skyc2, skyc3, skyc4, skyl2, skyl3, skyl4 | 30254561 | Preserved natural sparsity in sparse weather columns; no blanket imputation performed here. | These fields are sparse by nature, so missingness is meaningful. |

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
flight_number                                    float64
tail_number                                       string
destination_airport                               string
scheduled_departure_time                          string
actual_departure_time                             string
scheduled_elapsed_time_minutes                   float64
actual_elapsed_time_minutes                      float64
departure_delay_minutes                          float64
wheelsoff_time                                    string
taxiout_time_minutes                             float64
delay_carrier_minutes                            float64
delay_weather_minutes                            float64
delay_national_aviation_system_minutes           float64
delay_security_minutes                           float64
delay_late_aircraft_arrival_minutes              float64
origin_airport                                    string
airline                                           string
year_x                                           float64
date_dt                                   datetime64[us]
scheduled_departure_dt                    datetime64[us]
station                                           string
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
wxcodes                                           string
skyc1                                             string
skyc2                                             string
skyc3                                             string
skyc4                                             string
skyl1                                            float64
skyl2                                            float64
skyl3                                            float64
skyl4                                            float64
year_y                                           float64
carrier_name                                      string
carrier_type                                      string
hub_airports                                      string
airport_name                                      string
city                                              string
state                                             string
latitude                                         float64
longitude                                        float64
elevation_ft                                     float64
airport_type                                      string
num_runways                                        int64
timezone                                          string
departure_delayed                                   Int8
```

## Target Distribution

```
on_time    3824256
delayed    1152926
```