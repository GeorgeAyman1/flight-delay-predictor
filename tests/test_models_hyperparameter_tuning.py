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
@patch('src.models.hyperparameter_tuning.GridSearchCV')
@patch('builtins.open', new_callable=MagicMock)
def test_hyperparameter_tuner_run(
    mock_open, mock_search_cv, mock_pickle_dump, mock_json_load, mock_read_parquet, tmp_path
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
    mock_json_load.side_effect = [
        {"0": 1.0, "1": 1.5}, # for class weights
        {
            "subsample_freq": 1,
            "subsample": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.1,
            "n_estimators": 500,
            "max_depth": 8,
            "colsample_bytree": 0.7,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 100
        } # for stage1_best_params.json
    ]
    
    # Mock GridSearchCV
    mock_search = MagicMock()
    mock_search.best_params_ = {'learning_rate': 0.05}
    mock_search.best_score_ = 0.85
    mock_search_cv.return_value = mock_search
    
    tuner = HyperparameterTuner(base_dir=tmp_path, model_to_tune="lightgbm", n_iter=2)
    res = tuner.run()
    
    assert 'best_params' in res
    assert 'model_path' in res
    
    assert res['best_params']['learning_rate'] == 0.05
    
    assert mock_search_cv.call_count == 1
    mock_search.fit.assert_called_once()
    
    assert mock_pickle_dump.call_count == 1
