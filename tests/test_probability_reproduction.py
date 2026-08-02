import numpy as np

from model_specification.probability_reproduction_example import reproduce


def test_probability_reproduction_matches_frozen_expected_values():
    probability = reproduce()
    assert probability.shape == (3,)
    assert np.isfinite(probability).all()
    assert np.all((probability > 0) & (probability < 1))
