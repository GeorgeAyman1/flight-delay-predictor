import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.model_training import ModelTrainer

@patch('src.models.model_training.pd.read_parquet')
@patch('src.models.model_training.json.load')
@patch('src.models.model_training.pickle.dump')
@patch('src.models.model_training.json.dump')
@patch('builtins.open', new_callable=MagicMock)
def test_model_trainer_run(mock_open, mock_json_dump, mock_pickle_dump, mock_json_load, mock_read_parquet, tmp_path):
    n_samples = 20
    
    train_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'departure_delayed': np.random.randint(0, 2, n_samples)
    })
    
    valid_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'departure_delayed': np.random.randint(0, 2, n_samples)
    })
    
    mock_read_parquet.side_effect = [train_df, valid_df]
    mock_json_load.return_value = {"0": 1.0, "1": 1.5}
    
    trainer = ModelTrainer(base_dir=tmp_path)
    res = trainer.run()
    
    assert 'train_times' in res
    assert 'models_saved' in res
    
    # 7 base models + optional CatBoost
    assert len(res['train_times']) >= 7
    assert len(res['models_saved']) >= 7
    
    assert 'logistic_regression' in res['train_times']
    assert 'random_forest' in res['train_times']
    assert 'extra_trees' in res['train_times']
    assert 'xgboost' in res['train_times']
    assert 'lightgbm' in res['train_times']
    assert 'hist_gradient_boosting' in res['train_times']
    assert 'gradient_boosting' in res['train_times']
    
    # 1 scaler + N models
    assert mock_pickle_dump.call_count >= 8
