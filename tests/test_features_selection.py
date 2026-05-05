import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch

from src.features.feature_selection import FeatureSelector

@patch('src.features.feature_selection.pd.read_parquet')
@patch('pandas.DataFrame.to_parquet')
def test_feature_selector_run(mock_to_parquet, mock_read_parquet):
    # Craft a dataframe that will pass through all the filters
    # 1. Non-numeric columns (will be dropped)
    # 2. Low variance column (will be dropped, < 1% variance)
    # 3. High correlation pair (one will be dropped)
    # 4. Low MI (random noise vs target)
    # 5. Low importance (another random noise)
    
    np.random.seed(42)
    n_samples = 200
    
    # Target
    y = np.random.randint(0, 2, n_samples)
    
    # Good feature
    good_feat = y.copy().astype(float)
    
    # Correlated with good feature
    corr_feat = good_feat.copy()
    
    # Low variance feature
    low_var = np.ones(n_samples)
    low_var[0] = 0 # tiny variance
    
    # Random noise (should have low MI / Importance)
    noise = np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame({
        'non_numeric': ['A'] * n_samples,
        'good_feat': good_feat,
        'corr_feat': corr_feat,
        'low_var': low_var,
        'noise': noise,
        'day_of_week': np.random.randint(0, 7, n_samples), # force keep
        'latitude': np.random.uniform(30, 45, n_samples), # force drop
        'departure_delayed': y
    })
    
    mock_read_parquet.return_value = df
    
    selector = FeatureSelector(Path('/fake/path'))
    
    # Adjust thresholds to ensure our mock data works without brittle failures
    selector.VARIANCE_THRESHOLD = 0.05
    selector.CORRELATION_THRESHOLD = 0.95
    selector.MI_THRESHOLD = 0.0001
    selector.IMPORTANCE_THRESHOLD = 0.05
    selector.SAMPLE_FRACTION = 1.0 # use all for testing
    
    res = selector.run()
    
    assert 'selected_features' in res
    assert 'n_dropped' in res
    
    selected = res['selected_features']
    
    # non_numeric should be dropped
    assert 'non_numeric' not in selected
    # low_var should be dropped
    assert 'low_var' not in selected
    # force drop should be dropped
    assert 'latitude' not in selected
    # force keep should be kept
    assert 'day_of_week' in selected
    
    # One of the correlated features should be dropped
    assert not ('good_feat' in selected and 'corr_feat' in selected)
    
    assert mock_read_parquet.call_count == 3
    assert mock_to_parquet.call_count == 3
