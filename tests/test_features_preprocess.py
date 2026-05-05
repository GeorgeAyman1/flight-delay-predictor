import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.features.preprocess import (
    SkyC1Encoder,
    IsGustyTransformer,
    NumCloudLayersTransformer,
    CloudCeilingTransformer,
    WxCodeTransformer,
    Preprocessor,
    build_preprocessor
)

def test_skyc1_encoder():
    df = pd.DataFrame({'skyc1': ['CLR', 'FEW', 'SCT', 'BKN', 'OVC', 'UNKNOWN', None]})
    encoder = SkyC1Encoder()
    res = encoder.transform(df)
    
    expected = [0, 1, 2, 3, 4, 5, 5]
    assert res['skyc1_encoded'].tolist() == expected

def test_is_gusty_transformer():
    df = pd.DataFrame({'gust': [None, 20.5, np.nan, 15.0]})
    transformer = IsGustyTransformer()
    res = transformer.transform(df)
    
    assert res['is_gusty'].tolist() == [0, 1, 0, 1]
    assert res['gust'].tolist() == [0.0, 20.5, 0.0, 15.0]

def test_num_cloud_layers_transformer():
    df = pd.DataFrame({
        'skyc2': ['SCT', None, 'FEW'],
        'skyc3': ['BKN', None, None],
        'skyc4': ['OVC', None, None]
    })
    transformer = NumCloudLayersTransformer()
    res = transformer.transform(df)
    
    assert res['num_cloud_layers'].tolist() == [3, 0, 1]

def test_cloud_ceiling_transformer():
    df = pd.DataFrame({
        'skyl1': [1000, None, 500, None],
        'skyl2': [2000, 3000, 400, None],
        'skyl3': [3000, None, None, None],
        'skyl4': [4000, None, None, None]
    })
    transformer = CloudCeilingTransformer()
    res = transformer.transform(df)
    
    assert res['cloud_ceiling'].tolist() == [1000.0, 3000.0, 400.0, 99999.0]

def test_wx_code_transformer():
    df = pd.DataFrame({
        'wxcodes': ['-RA FG', 'TSRA', 'FZSN', 'HZ', None]
    })
    transformer = WxCodeTransformer()
    res = transformer.transform(df)
    
    assert res['has_fog'].tolist() == [1, 0, 0, 0, 0]
    assert res['has_thunder'].tolist() == [0, 1, 0, 0, 0]
    assert res['has_rain'].tolist() == [1, 1, 0, 0, 0]
    assert res['has_snow'].tolist() == [0, 0, 1, 0, 0]
    assert res['has_freezing'].tolist() == [0, 0, 1, 0, 0]

def test_build_preprocessor():
    prep = build_preprocessor()
    assert prep is not None
    assert prep.get_params()['remainder'] == 'passthrough'

@patch('src.features.preprocess.pd.read_parquet')
@patch('src.features.preprocess.joblib.dump')
@patch('pandas.DataFrame.to_parquet')
def test_preprocessor_run(mock_to_parquet, mock_joblib_dump, mock_read_parquet):
    # Setup mock data
    mock_df = pd.DataFrame({
        'tmpf': [70.0, 65.0],
        'dwpf': [50.0, 45.0],
        'relh': [50.0, 40.0],
        'sknt': [10.0, 5.0],
        'vsby': [10.0, 10.0],
        'alti': [29.92, 30.00],
        'mslp': [1012.0, 1015.0],
        'feel': [70.0, 65.0],
        'p01i': [0.0, 0.1],
        'gust': [None, 20.0],
        'skyc1': ['CLR', 'OVC'],
        'skyc2': [None, 'BKN'],
        'skyc3': [None, None],
        'skyc4': [None, None],
        'skyl1': [None, 1000.0],
        'skyl2': [None, 2000.0],
        'skyl3': [None, None],
        'skyl4': [None, None],
        'wxcodes': [None, '-RA FG'],
        'drct': [180.0, 90.0],
        'departure_delayed': [0, 1]
    })
    mock_read_parquet.return_value = mock_df
    
    # Initialize Preprocessor
    base_dir = Path('/fake/path')
    prep = Preprocessor(base_dir)
    
    # Run
    res = prep.run()
    
    # Assertions
    assert mock_read_parquet.call_count == 3
    assert mock_to_parquet.call_count == 3
    assert mock_joblib_dump.call_count == 1
    
    assert 'train_shape' in res
    assert 'valid_shape' in res
    assert 'test_shape' in res
    assert 'preprocessor_path' in res
    
    assert res['train_shape'] == (2, 20)  # 20 columns after transformations and drop
