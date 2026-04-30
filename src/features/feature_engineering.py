import pandas as pd
from pathlib import Path
import holidays
from datetime import timedelta


class FeatureEngineer:
    """
    Applies all feature engineering steps to the preprocessed data splits:
    column drops, datetime extraction, time-of-day bins, holiday flags,
    historical aggregation rates, lag features, weather severity, and
    route congestion.

    Saves the engineered feature files to data/processed/.
    """

    TARGET = "departure_delayed"

    # ── Columns to drop (leakage / redundant / IDs) ──────────────
    COLS_TO_DROP = [
        # Leakage — only known after departure
        "actual_departure_time",
        "actual_elapsed_time_minutes",
        "wheelsoff_time",
        "taxiout_time_minutes",
        "departure_delay_minutes",
        "delay_carrier_minutes",
        "delay_weather_minutes",
        "delay_national_aviation_system_minutes",
        "delay_security_minutes",
        "delay_late_aircraft_arrival_minutes",
        # Duplicates / redundant
        "year_y", "lat", "lon",
        "elevation", "elevation_ft",
        "date_mmddyyyy", "alti",
        # Pure IDs / merge artifacts
        "flight_number", "station",
        "valid", "hub_airports",
        # Redundant with carrier_code / origin_airport
        "airline", "carrier_name",
        "city", "state", "airport_name",
    ]

    # ── Major holiday keywords for travel-window feature ─────────
    MAJOR_HOLIDAY_KEYWORDS = [
        "Christmas Day",
        "Thanksgiving",
        "Independence Day",
        "New Year's Day",
        "Memorial Day",
        "Labor Day",
    ]

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "processed"

    # ─────────────────────────────────────────────────────────────
    # PRIVATE HELPERS (one per feature engineering step)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _add_time_features(df):
        """
        Extract hour, day-of-week, and time-of-day bins from scheduled departure.

        Bin boundaries justified by delay rate analysis on train:
          late night  (23, 0–4) : ~22% — carry-over delays from previous day
          early morning (5–11)  : ~13% — lowest, fresh start
          afternoon   (12–16)   : ~25% — building congestion
          evening     (17–22)   : ~30% — peak delays
        """
        df['hour'] = df['scheduled_departure_dt'].dt.hour
        df['day_of_week'] = df['scheduled_departure_dt'].dt.dayofweek

        # Remap hour 23 to -1 so it falls into late night bin naturally
        df['hour'] = df['hour'].replace(23, -1)

        bins = [-2, 4, 11, 16, 22]
        labels = ['late_night', 'early_morning', 'afternoon', 'evening']
        df['time_of_day'] = pd.cut(df['hour'], bins=bins, labels=labels)

        dummies = pd.get_dummies(df['time_of_day'], prefix='tod', dtype=int)
        df = pd.concat([df, dummies], axis=1)

        df.drop(columns=['hour', 'time_of_day'], inplace=True)
        return df

    def _add_holiday_features(self, df, us_holidays, all_holiday_dates, window_dates):
        """Add is_holiday and is_holiday_window binary flags."""
        flight_dates = df['date_dt'].dt.date
        df['is_holiday'] = flight_dates.isin(all_holiday_dates).astype(int)
        df['is_holiday_window'] = flight_dates.isin(window_dates).astype(int)
        return df

    @staticmethod
    def _add_historical_features(train_df, *dfs):
        """
        Compute airline/airport/route delay rates from train only, then map
        to all splits. Routes with < 30 flights fall back to global mean.
        """
        airline_rates = train_df.groupby('carrier_code')['departure_delayed'].mean()
        airport_rates = train_df.groupby('origin_airport')['departure_delayed'].mean()

        train_df['route'] = train_df['origin_airport'] + '_' + train_df['destination_airport']
        route_counts = train_df.groupby('route').size()
        route_rates = train_df.groupby('route')['departure_delayed'].mean()

        MIN_SUPPORT = 30
        reliable_routes = route_counts[route_counts >= MIN_SUPPORT].index
        route_rates = route_rates[reliable_routes]

        global_mean = float(train_df['departure_delayed'].mean())

        print("\nRoute support filter:")
        print(f"  Total routes:    {len(route_counts)}")
        print(f"  Reliable routes: {len(reliable_routes)}")
        print(f"  Filtered out:    {len(route_counts) - len(reliable_routes)}")

        for df in [train_df, *dfs]:
            df['route'] = df['origin_airport'] + '_' + df['destination_airport']
            df['airline_delay_rate'] = df['carrier_code'].map(airline_rates).fillna(global_mean)
            df['airport_delay_rate'] = df['origin_airport'].map(airport_rates).fillna(global_mean)
            df['route_delay_rate'] = df['route'].map(route_rates).fillna(global_mean)
            df.drop(columns=['route'], inplace=True)

        print("\nAirline delay rates (train only):")
        print(airline_rates.round(3).to_string())
        print("\nAirport delay rates (train only):")
        print(airport_rates.round(3).to_string())
        print("\nNaN check after mapping:")
        print(train_df[['airline_delay_rate', 'airport_delay_rate', 'route_delay_rate']].isna().sum().to_string())

        return train_df

    @staticmethod
    def _add_lag_feature(df):
        """
        Capture late-aircraft propagation via prev_flight_delayed.
        Uses a 6-hour gap threshold — beyond 6 hours the plane has been
        serviced and reset. Applied per split independently (no cross-split leakage).
        """
        df['tail_number'] = df['tail_number'].fillna('UNKNOWN')
        df = df.sort_values(['tail_number', 'scheduled_departure_dt'])

        df['prev_dep_time'] = df.groupby('tail_number')['scheduled_departure_dt'].shift(1)
        df['time_gap_hours'] = (
            df['scheduled_departure_dt'] - df['prev_dep_time']
        ).dt.total_seconds() / 3600

        df['prev_flight_delayed'] = (
            df.groupby('tail_number')['departure_delayed']
            .transform(lambda x: x.astype(float).shift(1))
        )

        df['prev_flight_delayed'] = df['prev_flight_delayed'].fillna(0).astype(int)
        df.loc[df['time_gap_hours'] > 6, 'prev_flight_delayed'] = 0
        df.loc[df['tail_number'] == 'UNKNOWN', 'prev_flight_delayed'] = 0

        df.drop(columns=['prev_dep_time', 'time_gap_hours'], inplace=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def _add_weather_severity(df, split_name='train'):
        """
        Combine multiple weather conditions into one weighted severity score.
        Weights reflect danger level for flight operations.
        """
        vis_normalized = (1 / (df['vsby'] + 1e-6)).clip(upper=10) / 2

        df['weather_severity'] = (
            df['has_thunder'] * 5 +
            df['has_fog'] * 4 +
            df['has_freezing'] * 3 +
            df['has_snow'] * 3 +
            df['has_rain'] * 2 +
            df['skyc1_encoded'] * 1 +
            vis_normalized * 1
        )

        if split_name == 'train':
            delayed_avg = df[df['departure_delayed'] == 1]['weather_severity'].mean()
            ontime_avg = df[df['departure_delayed'] == 0]['weather_severity'].mean()
            print("\nWeather severity signal check (train only):")
            print(f"  Delayed avg:    {delayed_avg:.3f}")
            print(f"  On-time avg:    {ontime_avg:.3f}")
            print(f"  Signal correct: {delayed_avg > ontime_avg} ← delayed should be higher")

        print(f"\n  [{split_name}] weather_severity — min: {df['weather_severity'].min():.3f} "
              f"| max: {df['weather_severity'].max():.3f} "
              f"| mean: {df['weather_severity'].mean():.3f} "
              f"| NaN: {df['weather_severity'].isna().sum()}")

        return df

    @staticmethod
    def _add_route_congestion(train_df, *dfs):
        """
        Capture how congested a route is relative to airport runway capacity.
        Metric: route_frequency / num_runways. Binned into low/medium/high
        using train percentiles (33rd, 66th). Routes with < 30 flights are
        unreliable → treated as unknown → fillna(0).
        """
        train_df['route'] = train_df['origin_airport'] + '_' + train_df['destination_airport']
        route_counts = train_df.groupby('route').size()

        MIN_SUPPORT = 30
        reliable_routes = route_counts[route_counts >= MIN_SUPPORT].index
        route_counts = route_counts[reliable_routes]

        train_df['route_frequency'] = train_df['route'].map(route_counts).fillna(0)
        train_df['num_runways_safe'] = train_df['num_runways'].replace(0, 1)
        train_df['route_congestion_raw'] = (
            train_df['route_frequency'] / train_df['num_runways_safe']
        )

        p33 = train_df['route_congestion_raw'].quantile(0.33)
        p66 = train_df['route_congestion_raw'].quantile(0.66)

        print("\nRoute congestion thresholds (train only):")
        print(f"  33rd percentile: {p33:.1f} flights/runway → low/medium boundary")
        print(f"  66th percentile: {p66:.1f} flights/runway → medium/high boundary")

        def assign_congestion(df):
            df['route'] = df['origin_airport'] + '_' + df['destination_airport']
            df['route_frequency'] = df['route'].map(route_counts).fillna(0)
            df['num_runways_safe'] = df['num_runways'].replace(0, 1)
            df['route_congestion_raw'] = df['route_frequency'] / df['num_runways_safe']

            df['route_congestion'] = 0
            df.loc[df['route_congestion_raw'] >= p33, 'route_congestion'] = 1
            df.loc[df['route_congestion_raw'] >= p66, 'route_congestion'] = 2

            df.drop(columns=['route', 'route_frequency',
                             'num_runways_safe', 'route_congestion_raw'], inplace=True)
            return df

        train_df = assign_congestion(train_df)
        for df in dfs:
            assign_congestion(df)

        print("\nRoute congestion value counts (train):")
        print(train_df['route_congestion'].value_counts().sort_index().to_string())
        print(f"  Unique values: {sorted(train_df['route_congestion'].unique())}")
        print(f"  NaN check:     {train_df['route_congestion'].isna().sum()}")
        print("\nDelay rate by congestion level (train only):")
        print(train_df.groupby('route_congestion')['departure_delayed'].mean().round(3).to_string())

        return train_df

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Apply all feature engineering steps and save outputs.

        Returns
        -------
        dict with keys:
            - train_shape : tuple
            - valid_shape : tuple
            - test_shape  : tuple
        """
        # ── Load preprocessed splits ─────────────────────────────
        train_pre = pd.read_parquet(self.data_dir / "train_preprocessed.parquet")
        valid_pre = pd.read_parquet(self.data_dir / "valid_preprocessed.parquet")
        test_pre = pd.read_parquet(self.data_dir / "test_preprocessed.parquet")

        print("Loaded shapes:")
        print(f"  Train: {train_pre.shape}")
        print(f"  Valid: {valid_pre.shape}")
        print(f"  Test:  {test_pre.shape}")

        train = train_pre.copy()
        valid = valid_pre.copy()
        test = test_pre.copy()

        # ── STEP 1 — DROP COLUMNS ────────────────────────────────
        for df in [train, valid, test]:
            df.drop(columns=self.COLS_TO_DROP, inplace=True, errors='ignore')

        print("\nAfter dropping leakage/redundant columns:")
        print(f"  Train columns: {train.shape[1]}")

        # ── STEP 2 — FIX DATETIME COLUMNS ───────────────────────
        for df in [train, valid, test]:
            df['scheduled_departure_time'] = pd.to_datetime(df['scheduled_departure_time'], format='mixed')
            df['date_dt'] = pd.to_datetime(df['date_dt'], format='mixed')
            df['scheduled_departure_dt'] = pd.to_datetime(df['scheduled_departure_dt'], format='mixed')

        print("\nDatetime columns converted successfully")

        # ── STEP 3 — TIME FEATURES ──────────────────────────────
        train = self._add_time_features(train)
        valid = self._add_time_features(valid)
        test = self._add_time_features(test)

        print("\nAfter time features:")
        print(f"  Train shape: {train.shape}")
        print(f"  New columns: {[c for c in train.columns if c.startswith('tod') or c == 'day_of_week']}")

        # ── STEP 4 — HOLIDAY FEATURES ───────────────────────────
        us_holidays_obj = holidays.US(years=[2022, 2023, 2024, 2025])
        all_holiday_dates = set(us_holidays_obj.keys())

        major_dates = {
            date for date, name in us_holidays_obj.items()
            if any(kw in name for kw in self.MAJOR_HOLIDAY_KEYWORDS)
        }
        window_dates = set()
        for d in major_dates:
            for delta in range(-2, 3):
                window_dates.add(d + timedelta(days=delta))

        train = self._add_holiday_features(train, us_holidays_obj, all_holiday_dates, window_dates)
        valid = self._add_holiday_features(valid, us_holidays_obj, all_holiday_dates, window_dates)
        test = self._add_holiday_features(test, us_holidays_obj, all_holiday_dates, window_dates)

        print("\nAfter holiday features:")
        print(f"  Train shape: {train.shape}")
        print(f"  is_holiday count:        {train['is_holiday'].sum()}")
        print(f"  is_holiday_window count: {train['is_holiday_window'].sum()}")

        # ── STEP 5 — HISTORICAL AGGREGATION FEATURES ────────────
        train = self._add_historical_features(train, valid, test)

        print("\nAfter historical features:")
        print(f"  Train shape: {train.shape}")

        # ── STEP 6 — LAG FEATURE ────────────────────────────────
        train = self._add_lag_feature(train)
        valid = self._add_lag_feature(valid)
        test = self._add_lag_feature(test)

        print("\nAfter lag feature:")
        print(f"  Train shape: {train.shape}")
        print("  prev_flight_delayed value counts:")
        print(train['prev_flight_delayed'].value_counts().to_string())
        print(f"  NaN check: {train['prev_flight_delayed'].isna().sum()}")

        # ── STEP 7 — WEATHER SEVERITY SCORE ─────────────────────
        train = self._add_weather_severity(train, split_name='train')
        valid = self._add_weather_severity(valid, split_name='valid')
        test = self._add_weather_severity(test, split_name='test')

        print("\nAfter weather severity:")
        print(f"  Train shape: {train.shape}")

        # ── STEP 8 — ROUTE CONGESTION ───────────────────────────
        train = self._add_route_congestion(train, valid, test)

        print("\nAfter route congestion:")
        print(f"  Train shape: {train.shape}")

        # ── STEP 9 — FINAL VALIDATION BEFORE SAVING ─────────────
        print(f"\n{'═'*60}")
        print("FINAL VALIDATION")
        print(f"{'═'*60}")

        expected_new_cols = [
            'day_of_week', 'tod_late_night', 'tod_early_morning', 'tod_afternoon', 'tod_evening',
            'is_holiday', 'is_holiday_window',
            'airline_delay_rate', 'airport_delay_rate', 'route_delay_rate',
            'prev_flight_delayed',
            'weather_severity',
            'route_congestion',
        ]

        print("\nExpected new feature columns:")
        for col in expected_new_cols:
            in_train = col in train.columns
            in_valid = col in valid.columns
            in_test = col in test.columns
            status = '✓' if (in_train and in_valid and in_test) else '✗ MISSING'
            print(f"  {status}  {col}")

        print("\nFinal shapes:")
        print(f"  Train: {train.shape}")
        print(f"  Valid: {valid.shape}")
        print(f"  Test:  {test.shape}")

        print("\nNaN summary (train):")
        nan_cols = train.isnull().sum()
        nan_cols = nan_cols[nan_cols > 0]
        if len(nan_cols) == 0:
            print("  No NaNs found ✓")
        else:
            print(nan_cols.to_string())

        print("\ndeparture_delayed still present:")
        print(f"  Train: {'departure_delayed' in train.columns} ✓")
        print(f"  Valid: {'departure_delayed' in valid.columns} ✓")
        print(f"  Test:  {'departure_delayed' in test.columns} ✓")

        # ── STEP 10 — SAVE OUTPUTS ──────────────────────────────
        train.to_parquet(self.data_dir / "train_features.parquet", index=False)
        valid.to_parquet(self.data_dir / "valid_features.parquet", index=False)
        test.to_parquet(self.data_dir / "test_features.parquet", index=False)

        print(f"\n{'═'*60}")
        print("Saved successfully:")
        print(f"  data/processed/train_features.parquet  ({train.shape[0]:,} rows, {train.shape[1]} cols)")
        print(f"  data/processed/valid_features.parquet  ({valid.shape[0]:,} rows, {valid.shape[1]} cols)")
        print(f"  data/processed/test_features.parquet   ({test.shape[0]:,} rows, {test.shape[1]} cols)")
        print(f"{'═'*60}")

        return {
            "train_shape": train.shape,
            "valid_shape": valid.shape,
            "test_shape": test.shape,
        }


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    engineer = FeatureEngineer(base)
    engineer.run()