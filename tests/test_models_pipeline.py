from unittest.mock import patch, MagicMock

from src.models.model_pipeline import ModelPipeline

@patch('src.models.model_pipeline.ClassWeightCalculator')
@patch('src.models.model_pipeline.ModelTrainer')
@patch('src.models.model_pipeline.ModelComparator')
@patch('src.models.model_pipeline.WeightComparisonAnalyzer')
@patch('src.models.model_pipeline.HyperparameterTuner')
def test_model_pipeline_run(
    mock_tuner_class, mock_analyzer_class, mock_comparator_class,
    mock_trainer_class, mock_weight_class
):
    mock_weight = MagicMock()
    mock_weight.run.return_value = {'weights': 'done'}
    mock_weight_class.return_value = mock_weight
    
    mock_trainer = MagicMock()
    mock_trainer.run.return_value = {'training': 'done'}
    mock_trainer_class.return_value = mock_trainer
    
    mock_comparator = MagicMock()
    mock_comparator.run.return_value = {'winner': 'xgboost'}
    mock_comparator_class.return_value = mock_comparator
    
    mock_analyzer = MagicMock()
    mock_analyzer.run.return_value = {'weight_comp': 'done'}
    mock_analyzer_class.return_value = mock_analyzer
    
    mock_tuner = MagicMock()
    mock_tuner.run.return_value = {'model_path': '/fake/path'}
    mock_tuner_class.return_value = mock_tuner
    
    pipeline = ModelPipeline(include_weight_comparison=True)
    res = pipeline.run()
    
    mock_weight_class.assert_called_once()
    mock_weight.run.assert_called_once()
    
    mock_trainer_class.assert_called_once()
    mock_trainer.run.assert_called_once()
    
    mock_comparator_class.assert_called_once()
    mock_comparator.run.assert_called_once()
    
    mock_analyzer_class.assert_called_once()
    mock_analyzer.run.assert_called_once()
    
    mock_tuner_class.assert_called_once()
    mock_tuner.run.assert_called_once()
    
    assert 'class_weights' in res
    assert 'training' in res
    assert 'comparison' in res
    assert 'weight_comparison' in res
    assert 'tuning' in res
