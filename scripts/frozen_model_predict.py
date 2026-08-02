"""Calculate frozen LASSO probabilities from public, data-free parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ORDER = PROJECT / "model_specification" / "final_feature_order.csv"
DEFAULT_PREPROCESSING = PROJECT / "model_specification" / "preprocessing_parameters.csv"
DEFAULT_METADATA = PROJECT / "model_specification" / "model_metadata.json"


def _binary(series: pd.Series) -> np.ndarray:
    lowered = series.astype(str).str.lower()
    valid = lowered.isin(["true", "t", "1", "false", "f", "0"])
    if not valid.all():
        raise ValueError("Binary predictors must be encoded as true/false or 1/0 and may not be missing")
    return lowered.isin(["true", "t", "1"]).astype(float).to_numpy()


def transformed_matrix(
    frame: pd.DataFrame, feature_order: pd.DataFrame, preprocessing: pd.DataFrame
) -> np.ndarray:
    transformed: dict[str, np.ndarray] = {}
    for _, row in preprocessing.loc[preprocessing["component_type"] == "numeric"].iterrows():
        feature = row["original_variable"]
        values = pd.to_numeric(frame[feature], errors="coerce")
        missing = values.isna().astype(float).to_numpy()
        imputed = values.fillna(float(row["development_imputation_median"])).to_numpy(dtype=float)
        transformed[row["transformed_feature"]] = (
            imputed - float(row["scaler_mean"])
        ) / float(row["scaler_scale"])
        if str(row["missing_indicator_generated"]).lower() == "true":
            transformed[row["missing_indicator_transformed_feature"]] = (
                missing - float(row["missing_indicator_scaler_mean"])
            ) / float(row["missing_indicator_scaler_scale"])

    if frame["gender"].isna().any():
        raise ValueError("The public specification does not impute missing gender; SQL-derived gender is required")
    gender = frame["gender"].astype(str).str.upper()
    transformed["categorical__gender_F"] = gender.eq("F").astype(float).to_numpy()
    transformed["categorical__gender_M"] = gender.eq("M").astype(float).to_numpy()

    binary_rows = preprocessing.loc[preprocessing["component_type"] == "binary"]
    for _, row in binary_rows.iterrows():
        transformed[row["transformed_feature"]] = _binary(frame[row["original_variable"]])

    ordered = feature_order.sort_values("transform_order")
    missing_transformed = sorted(set(ordered["transformed_feature"]) - set(transformed))
    if missing_transformed:
        raise ValueError(f"Unable to construct transformed features: {missing_transformed}")
    return np.column_stack([transformed[name] for name in ordered["transformed_feature"]])


def predict_frozen(frame: pd.DataFrame, feature_order: pd.DataFrame, preprocessing: pd.DataFrame,
                   intercept: float = -3.707283461539647) -> np.ndarray:
    ordered = feature_order.sort_values("transform_order")
    matrix = transformed_matrix(frame, ordered, preprocessing)
    coefficients = ordered["coefficient"].to_numpy(dtype=float)
    if matrix.shape[1] != 41 or len(coefficients) != 41:
        raise ValueError("Frozen model must contain exactly 41 transformed inputs")
    return expit(intercept + matrix @ coefficients)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen data-free LASSO specification")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-order", type=Path, default=DEFAULT_ORDER)
    parser.add_argument("--preprocessing", type=Path, default=DEFAULT_PREPROCESSING)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    frame = pd.read_csv(args.input)
    order = pd.read_csv(args.feature_order)
    preprocessing = pd.read_csv(args.preprocessing)
    result = frame.copy()
    result["predicted_risk"] = predict_frozen(frame, order, preprocessing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} local predictions to {args.output}")


if __name__ == "__main__":
    main()
