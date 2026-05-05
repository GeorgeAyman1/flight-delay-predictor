import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.model_comparison import ModelComparator

@patch('src.models.model_comparison.pd.read_parquet')
@patch('src.models.model_comparison.pickle.load')
@patch('src.models.model_comparison.Path.exists')
@patch('src.models.model_comparison.mlflow')
@patch('src.models.model_comparison.MlflowClient')
@patch('src.models.model_comparison.mlflow.set_tracking_uri')
@patch('src.models.model_comparison.plt')
@patch('builtins.open', new_callable=MagicMock)
def test_model_comparator_run(
    mock_open, mock_plt, mock_set_uri, mock_mlflow_client, mock_mlflow, 
    mock_path_exists, mock_pickle_load, mock_read_parquet, tmp_path
):
    n_samples = 20
    
    valid_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'departure_delayed': np.random.randint(0, 2, n_samples)
    })
    
    mock_read_parquet.return_value = valid_df
    
    # Path.exists() for checking if model exists
    mock_path_exists.return_value = True
    
    # Mock scaler and models for pickle.load
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((n_samples, 1))
    
    mock_model = MagicMock()
    mock_model.predict.return_value = np.random.randint(0, 2, n_samples)
    mock_model.predict_proba.return_value = np.random.uniform(0, 1, (n_samples, 2))
    mock_model.get_params.return_value = {'param1': 1}
    
    # Side effect for pickle.load to return scaler first, then models
    mock_pickle_load.side_effect = [mock_scaler, mock_model, mock_model, mock_model, mock_model, mock_model]
    
    # Mock subplots
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    
    comparator = ModelComparator(base_dir=tmp_path)
    res = comparator.run()
    
    assert 'results_df' in res
    assert 'winner' in res
    
    assert len(res['results_df']) == 5
    assert mock_pickle_load.call_count == 6 # scaler + 5 models
