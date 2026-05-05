import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.models.model_weight_comparison import WeightComparisonAnalyzer

@patch('src.models.model_weight_comparison.pd.read_parquet')
@patch('src.models.model_weight_comparison.json.load')
@patch('builtins.open', new_callable=MagicMock)
def test_weight_comparison_analyzer_run(mock_open, mock_json_load, mock_read_parquet, tmp_path):
    n_samples = 40
    
    train_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'departure_delayed': np.random.randint(0, 2, n_samples)
    })
    
    valid_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'departure_delayed': np.random.randint(0, 2, n_samples)
    })
    
    mock_read_parquet.side_effect = [train_df, valid_df]
    mock_json_load.return_value = {"0": 1.0, "1": 5.0}
    
    analyzer = WeightComparisonAnalyzer(base_dir=tmp_path)
    res = analyzer.run()
    
    assert 'unweighted' in res
    assert 'weighted' in res
    
    # Check metrics existence
    for k in ['roc_auc', 'recall', 'f1']:
        assert k in res['unweighted']
        assert k in res['weighted']
