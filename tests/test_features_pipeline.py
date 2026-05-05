import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.features.feature_pipeline import FeaturePipeline

@patch('src.features.feature_pipeline.Preprocessor')
@patch('src.features.feature_pipeline.FeatureEngineer')
@patch('src.features.feature_pipeline.FeatureSelector')
def test_feature_pipeline_run(mock_selector_class, mock_engineer_class, mock_preprocessor_class):
    # Setup mocks
    mock_preprocessor = MagicMock()
    mock_preprocessor.run.return_value = {'prep': 'done'}
    mock_preprocessor_class.return_value = mock_preprocessor
    
    mock_engineer = MagicMock()
    mock_engineer.run.return_value = {'eng': 'done'}
    mock_engineer_class.return_value = mock_engineer
    
    mock_selector = MagicMock()
    mock_selector.run.return_value = {
        'selected_features': ['A', 'B'],
        'train_shape': (10, 2),
        'valid_shape': (10, 2),
        'test_shape': (10, 2)
    }
    mock_selector_class.return_value = mock_selector
    
    pipeline = FeaturePipeline(base_dir=Path('/fake/path'))
    res = pipeline.run()
    
    # Verify calls
    mock_preprocessor_class.assert_called_once()
    mock_preprocessor.run.assert_called_once()
    
    mock_engineer_class.assert_called_once()
    mock_engineer.run.assert_called_once()
    
    mock_selector_class.assert_called_once()
    mock_selector.run.assert_called_once()
    
    # Verify result dictionary
    assert 'preprocessing' in res
    assert 'feature_engineering' in res
    assert 'feature_selection' in res
    
    assert res['preprocessing'] == {'prep': 'done'}
    assert res['feature_engineering'] == {'eng': 'done'}
