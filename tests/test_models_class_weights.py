import pandas as pd
import json
from unittest.mock import patch

from src.models.class_weights import ClassWeightCalculator

@patch('src.models.class_weights.pd.read_parquet')
def test_class_weights_calculator(mock_read_parquet, tmp_path):
    # Dummy data with 3 on-time (0) and 1 delayed (1) -> 4 total
    df = pd.DataFrame({'departure_delayed': [0, 0, 0, 1]})
    mock_read_parquet.return_value = df
    
    # We use tmp_path to write the actual file
    base_dir = tmp_path
    
    calculator = ClassWeightCalculator(base_dir=base_dir)
    res = calculator.run()
    
    # Expected balanced weights:
    # 0 (on-time): 4 / (2 * 3) = 4/6 = 0.666667
    # 1 (delayed): 4 / (2 * 1) = 4/2 = 2.0
    
    assert 'sklearn_weights' in res
    assert 'scale_pos_weight' in res
    
    weights = res['sklearn_weights']
    assert abs(weights[0] - 0.666667) < 1e-5
    assert abs(weights[1] - 2.0) < 1e-5
    
    assert abs(res['scale_pos_weight'] - 3.0) < 1e-5
    
    # Check if saved
    saved_file = base_dir / "models" / "class_weights.json"
    assert saved_file.exists()
    
    with open(saved_file) as f:
        saved_weights = json.load(f)
        assert saved_weights['0'] == weights[0]
        assert saved_weights['1'] == weights[1]
