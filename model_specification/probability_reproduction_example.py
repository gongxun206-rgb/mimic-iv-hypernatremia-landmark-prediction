"""Reproduce expected probabilities without loading a serialized model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.frozen_model_predict import predict_frozen  # noqa: E402


def reproduce() -> np.ndarray:
    example = pd.read_csv(ROOT / "synthetic_example" / "synthetic_example.csv")
    feature_order = pd.read_csv(ROOT / "model_specification" / "final_feature_order.csv")
    coefficients = pd.read_csv(ROOT / "model_specification" / "final_coefficients.csv")
    preprocessing = pd.read_csv(ROOT / "model_specification" / "preprocessing_parameters.csv")
    expected = json.loads(
        (ROOT / "model_specification" / "expected_synthetic_probability.json").read_text(encoding="utf-8")
    )
    coefficients = coefficients.sort_values("transform_order").reset_index(drop=True)
    feature_order = feature_order.sort_values("transform_order").reset_index(drop=True)
    if not feature_order["transformed_feature"].equals(coefficients["transformed_feature"]):
        raise ValueError("Feature order and coefficient specification are not aligned")
    feature_order["coefficient"] = coefficients["coefficient"].to_numpy()
    observed = predict_frozen(example, feature_order, preprocessing, float(coefficients.loc[0, "intercept"]))
    np.testing.assert_allclose(
        observed, np.asarray(expected["predicted_risk"]), rtol=0,
        atol=float(expected["absolute_tolerance"]),
    )
    return observed


if __name__ == "__main__":
    print("\n".join(f"{value:.17g}" for value in reproduce()))
