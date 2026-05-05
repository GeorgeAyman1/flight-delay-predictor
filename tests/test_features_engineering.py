import pandas as pd
from pathlib import Path
from datetime import date
from unittest.mock import patch

from src.features.feature_engineering import FeatureEngineer

def test_add_time_features():
    df = pd.DataFrame({
        'scheduled_departure_dt': pd.to_datetime([
            '2023-01-01 02:00:00', # late_night
            '2023-01-01 08:00:00', # early_morning
            '2023-01-01 14:00:00', # afternoon
            '2023-01-01 20:00:00', # evening
            '2023-01-01 23:00:00', # late_night (hour 23)
        ])
    })
    
    res = FeatureEngineer._add_time_features(df.copy())
    
    assert 'day_of_week' in res.columns
    assert res['tod_late_night'].tolist() == [1, 0, 0, 0, 1]
    assert res['tod_early_morning'].tolist() == [0, 1, 0, 0, 0]
    assert res['tod_afternoon'].tolist() == [0, 0, 1, 0, 0]
    assert res['tod_evening'].tolist() == [0, 0, 0, 1, 0]

def test_add_holiday_features():
    df = pd.DataFrame({
        'date_dt': pd.to_datetime([
            '2023-12-25', # Christmas (holiday)
            '2023-12-24', # Window
            '2023-12-10', # Not holiday or window
        ])
    })
    
    eng = FeatureEngineer(Path('.'))
    us_holidays = {date(2023, 12, 25): "Christmas Day"}
    all_holiday_dates = {date(2023, 12, 25)}
    window_dates = {date(2023, 12, 24), date(2023, 12, 25), date(2023, 12, 26)}
    
    res = eng._add_holiday_features(df.copy(), us_holidays, all_holiday_dates, window_dates)
    
    assert res['is_holiday'].tolist() == [1, 0, 0]
    assert res['is_holiday_window'].tolist() == [1, 1, 0]

def test_add_historical_features():
    train_df = pd.DataFrame({
        'carrier_code': ['DL', 'DL', 'AA', 'AA', 'AA'] * 10,
        'origin_airport': ['JFK', 'LAX', 'JFK', 'LAX', 'ORD'] * 10,
        'destination_airport': ['LAX', 'JFK', 'ORD', 'ORD', 'JFK'] * 10,
        'departure_delayed': [1, 0, 1, 0, 0] * 10
    })
    
    test_df = pd.DataFrame({
        'carrier_code': ['DL', 'WN'], # WN is unseen
        'origin_airport': ['JFK', 'MIA'], # MIA is unseen
        'destination_airport': ['LAX', 'DFW']
    })
    
    res_train = FeatureEngineer._add_historical_features(train_df.copy(), test_df.copy())
    
    assert 'airline_delay_rate' in res_train.columns
    assert 'airport_delay_rate' in res_train.columns
    assert 'route_delay_rate' in res_train.columns
    
    # Train test - 'DL' has 1 delay out of 2 flights = 0.5 rate
    assert res_train['airline_delay_rate'].iloc[0] == 0.5
    
def test_add_lag_feature():
    df = pd.DataFrame({
        'tail_number': ['N1', 'N1', 'N1', 'N2', 'UNKNOWN'],
        'scheduled_departure_dt': pd.to_datetime([
            '2023-01-01 10:00:00',
            '2023-01-01 12:00:00', # 2 hours gap (propagate)
            '2023-01-01 20:00:00', # 8 hours gap (reset)
            '2023-01-01 10:00:00', # First flight for N2
            '2023-01-01 10:00:00'  # Unknown tail
        ]),
        'departure_delayed': [1, 0, 1, 1, 1]
    })
    
    res = FeatureEngineer._add_lag_feature(df.copy())
    
    # Needs sorting internally, so let's match by tail_number and scheduled_departure_dt
    n1_12 = res[(res['tail_number'] == 'N1') & (res['scheduled_departure_dt'] == '2023-01-01 12:00:00')].iloc[0]
    n1_20 = res[(res['tail_number'] == 'N1') & (res['scheduled_departure_dt'] == '2023-01-01 20:00:00')].iloc[0]
    n2 = res[res['tail_number'] == 'N2'].iloc[0]
    unk = res[res['tail_number'] == 'UNKNOWN'].iloc[0]
    
    assert n1_12['prev_flight_delayed'] == 1 # Propagated
    assert n1_20['prev_flight_delayed'] == 0 # Gap > 6 hours
    assert n2['prev_flight_delayed'] == 0 # First flight
    assert unk['prev_flight_delayed'] == 0 # Unknown tail

def test_add_weather_severity():
    df = pd.DataFrame({
        'has_thunder': [1, 0],
        'has_fog': [0, 1],
        'has_freezing': [0, 0],
        'has_snow': [0, 0],
        'has_rain': [1, 0],
        'skyc1_encoded': [4, 0],
        'vsby': [5.0, 0.5],
        'departure_delayed': [1, 0]
    })
    
    res = FeatureEngineer._add_weather_severity(df.copy())
    assert 'weather_severity' in res.columns
    # Score 1: thunder(5) + rain(2) + skyc1(4) + vis_norm ~0 = 11
    # Score 2: fog(4) + vis_norm ~1 = 5

def test_add_route_congestion():
    train_df = pd.DataFrame({
        'origin_airport': ['A'] * 40 + ['B'] * 40 + ['C'] * 10,
        'destination_airport': ['B'] * 40 + ['C'] * 40 + ['A'] * 10,
        'num_runways': [2] * 40 + [1] * 40 + [1] * 10,
        'departure_delayed': [0] * 90
    })
    
    res = FeatureEngineer._add_route_congestion(train_df.copy())
    assert 'route_congestion' in res.columns
    # 'C_A' route has < 30 flights, will be filtered -> 0 congestion

@patch('src.features.feature_engineering.pd.read_parquet')
@patch('pandas.DataFrame.to_parquet')
def test_feature_engineer_run(mock_to_parquet, mock_read_parquet):
    df = pd.DataFrame({
        'actual_departure_time': [0, 0], # to drop
        'scheduled_departure_time': ['2023-01-01 10:00:00', '2023-01-01 12:00:00'],
        'date_dt': ['2023-01-01', '2023-01-01'],
        'scheduled_departure_dt': ['2023-01-01 10:00:00', '2023-01-01 12:00:00'],
        'carrier_code': ['DL', 'AA'],
        'origin_airport': ['JFK', 'LAX'],
        'destination_airport': ['LAX', 'JFK'],
        'tail_number': ['N1', 'N2'],
        'num_runways': [2, 3],
        'has_thunder': [0, 1],
        'has_fog': [0, 0],
        'has_freezing': [0, 0],
        'has_snow': [0, 0],
        'has_rain': [0, 1],
        'skyc1_encoded': [0, 4],
        'vsby': [10.0, 2.0],
        'departure_delayed': [0, 1]
    })
    
    mock_read_parquet.return_value = df
    
    eng = FeatureEngineer(Path('/fake/path'))
    res = eng.run()
    
    assert mock_read_parquet.call_count == 3
    assert mock_to_parquet.call_count == 3
    
    assert 'train_shape' in res
    assert 'valid_shape' in res
    assert 'test_shape' in res
