import pytest
import pandas as pd
from pathlib import Path

TRAIN_PATH = Path("data/processed/train_preprocessed.parquet")

pytestmark = pytest.mark.skipif(
    not TRAIN_PATH.exists(),
    reason="Preprocessed data not available in CI — run preprocess.py first"
)

def test_expected_columns_exist():
    df = pd.read_parquet(TRAIN_PATH)
    expected = [
        'tmpf', 'dwpf', 'relh', 'sknt', 'vsby', 'alti', 'mslp', 'feel',
        'p01i', 'is_gusty', 'gust', 'num_cloud_layers', 'cloud_ceiling',
        'skyc1_encoded', 'has_fog', 'has_thunder', 'has_rain', 'has_snow',
        'has_freezing', 'departure_delayed'
    ]
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"

def test_no_nulls_in_imputed_cols():
    df = pd.read_parquet(TRAIN_PATH)
    imputed = ['tmpf', 'dwpf', 'relh', 'sknt', 'vsby', 'alti', 'mslp', 'feel', 'p01i', 'gust']
    for col in imputed:
        assert df[col].isnull().sum() == 0, f"Unexpected nulls in {col}"

def test_binary_flags_are_0_or_1():
    df = pd.read_parquet(TRAIN_PATH)
    flags = ['is_gusty', 'has_fog', 'has_thunder', 'has_rain', 'has_snow', 'has_freezing']
    for col in flags:
        assert df[col].isin([0, 1]).all(), f"{col} contains values other than 0 and 1"

def test_skyc1_encoded_range():
    df = pd.read_parquet(TRAIN_PATH)
    assert df['skyc1_encoded'].between(0, 5).all(), "skyc1_encoded has values outside 0–5"

def test_correct_row_count():
    df = pd.read_parquet(TRAIN_PATH)
    assert len(df) == 2_454_517, f"Unexpected row count: {len(df)}"