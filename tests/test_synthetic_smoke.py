from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_synthetic_example_is_artificial_and_complete():
    data = pd.read_csv(ROOT / 'synthetic_example' / 'synthetic_example.csv')
    assert len(data) == 3
    assert 'stay_id' not in data.columns
    assert 'subject_id' not in data.columns
    assert set(data['outcome_na_ge_151']) <= {0, 1}
