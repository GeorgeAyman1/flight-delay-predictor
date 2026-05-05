import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.model_comparison import ModelComparator

@patch('src.models.model_comparison.pd.read_parquet')
@patch('src.models.model_comparison.pickle.load')
@patch('src.models.model_comparison.Path.exists')
@patch('src.models.model_comparison.json.load')
@patch('builtins.open', new_callable=MagicMock)
def test_model_comparator_run(
    mock_open, mock_json_load, mock_path_exists, mock_pickle_load, mock_read_parquet, tmp_path
):
    n_samples = 20
    
    valid_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'departure_delayed': np.random.randint(0, 2, n_samples)
    })
    
    mock_read_parquet.return_value = valid_df
    
    # Path.exists() for checking if model exists
    mock_path_exists.return_value = True
    
    mock_json_load.return_value = {"logistic_regression": 0.5}
    
    # Mock scaler and models for pickle.load
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((n_samples, 1))
    
    mock_model = MagicMock()
    mock_model.predict.return_value = np.random.randint(0, 2, n_samples)
    mock_model.predict_proba.return_value = np.random.uniform(0, 1, (n_samples, 2))
    mock_model.get_params.return_value = {'param1': 1}
    
    # Side effect for pickle.load to return scaler first, then models
    mock_pickle_load.side_effect = [mock_scaler, mock_model, mock_model, mock_model, mock_model, mock_model, mock_model, mock_model, mock_model, mock_model]
    
    comparator = ModelComparator(base_dir=tmp_path)
    res = comparator.run()
    
    assert 'results' in res
    assert 'winner' in res
    
    assert len(res['results']) >= 7
    assert mock_pickle_load.call_count >= 8 # scaler + 7+ models
