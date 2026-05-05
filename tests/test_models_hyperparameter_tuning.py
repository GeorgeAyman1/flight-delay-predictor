import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.hyperparameter_tuning import HyperparameterTuner

@patch('src.models.hyperparameter_tuning.pd.read_parquet')
@patch('src.models.hyperparameter_tuning.json.load')
@patch('src.models.hyperparameter_tuning.pickle.dump')
@patch('src.models.hyperparameter_tuning.json.dump')
@patch('src.models.hyperparameter_tuning.mlflow')
@patch('src.models.hyperparameter_tuning.plt')
@patch('src.models.hyperparameter_tuning.RandomizedSearchCV')
@patch('builtins.open', new_callable=MagicMock)
def test_hyperparameter_tuner_run(
    mock_open, mock_search_cv, mock_plt, mock_mlflow, mock_json_dump, 
    mock_pickle_dump, mock_json_load, mock_read_parquet, tmp_path
):
    n_samples = 20
    
    train_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'departure_delayed': np.random.randint(0, 2, n_samples)
    })
    
    valid_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'departure_delayed': np.random.randint(0, 2, n_samples)
    })
    
    mock_read_parquet.side_effect = [train_df, valid_df]
    mock_json_load.return_value = {"0": 1.0, "1": 1.5}
    
    # Mock plt.subplots
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    
    # Mock RandomizedSearchCV
    mock_search = MagicMock()
    mock_search.best_params_ = {'max_depth': 6}
    mock_search.best_score_ = 0.85
    mock_search.cv_results_ = {
        'params': [{'max_depth': 6}],
        'mean_test_score': [0.85],
        'rank_test_score': [1]
    }
    mock_search_cv.return_value = mock_search
    
    tuner = HyperparameterTuner(base_dir=tmp_path, model_to_tune="xgboost", n_iter=2)
    res = tuner.run()
    
    assert 'best_params' in res
    assert 'best_score' in res
    assert 'model_path' in res
    
    assert res['best_score'] == 0.85
    assert res['best_params'] == {'max_depth': 6}
    
    assert mock_search_cv.call_count == 1
    mock_search.fit.assert_called_once()
    
    assert mock_pickle_dump.call_count == 1
    assert mock_json_dump.call_count == 1
