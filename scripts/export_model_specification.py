"""Export and independently reproduce the frozen 27-feature LASSO pipeline without refitting."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data" / "primary_cohort_model_dataset_v1.0.csv"
FEATURE_MANIFEST_PATH = PROJECT / "modeling" / "v1.0_locked_primary" / "core_feature_manifest.csv"
RUN_CONFIG_PATH = PROJECT / "modeling" / "v1.0_locked_primary" / "run_config.json"
MODEL_PATH = PROJECT / "modeling" / "v1.0_locked_primary" / "lasso_final_development.joblib"
OUTPUT_DIR = PROJECT / "paper_submission_materials" / "v1.0" / "lasso_specification_v1.0"
SEED = 20260731


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def normalize_binary(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for feature in features:
        output[feature] = output[feature].astype(str).str.lower().isin(["true", "t", "1"]).astype(float)
    return output


def manual_transformed_matrix(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str],
    numeric_medians: np.ndarray,
    numeric_means: np.ndarray,
    numeric_scales: np.ndarray,
    indicator_positions: np.ndarray,
    category_modes: np.ndarray,
    categories: list[np.ndarray],
    binary_modes: np.ndarray,
) -> np.ndarray:
    numeric_raw = frame[numeric_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    numeric_missing = np.isnan(numeric_raw)
    numeric_imputed = np.where(numeric_missing, numeric_medians, numeric_raw)
    transformed = [(numeric_imputed - numeric_means[: len(numeric_features)]) / numeric_scales[: len(numeric_features)]]
    if len(indicator_positions):
        raw_indicators = numeric_missing[:, indicator_positions].astype(float)
        start = len(numeric_features)
        transformed.append((raw_indicators - numeric_means[start:]) / numeric_scales[start:])

    for position, feature in enumerate(categorical_features):
        values = frame[feature].astype(object).where(frame[feature].notna(), category_modes[position]).astype(str).to_numpy()
        transformed.append(np.column_stack([(values == str(category)).astype(float) for category in categories[position]]))

    binary_raw = frame[binary_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    binary_imputed = np.where(np.isnan(binary_raw), binary_modes, binary_raw)
    transformed.append(binary_imputed)
    return np.hstack(transformed)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    data = pd.read_csv(DATA_PATH)
    feature_manifest = pd.read_csv(FEATURE_MANIFEST_PATH)
    run_config = json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
    pipeline = joblib.load(MODEL_PATH)
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    features = feature_manifest["feature"].tolist()
    numeric_features = list(preprocess.transformers_[0][2])
    categorical_features = list(preprocess.transformers_[1][2])
    binary_features = list(preprocess.transformers_[2][2])
    if features != numeric_features + categorical_features + binary_features:
        raise ValueError("Feature order differs from frozen preprocessing pipeline")
    if len(features) != 27 or len(data) != 34913 or data["stay_id"].duplicated().any():
        raise ValueError("Locked input data integrity failed")
    if run_config["lasso_final_C"] != 0.3 or model.C != 0.3:
        raise ValueError("Final C is not the locked value 0.3")
    if model.solver != "saga" or model.l1_ratio != 1.0:
        raise ValueError("Frozen model does not match expected saga/l1_ratio=1.0 configuration")

    numeric_pipeline = preprocess.named_transformers_["numeric"]
    numeric_imputer = numeric_pipeline.named_steps["imputer"]
    numeric_scaler = numeric_pipeline.named_steps["scaler"]
    categorical_pipeline = preprocess.named_transformers_["categorical"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    onehot = categorical_pipeline.named_steps["onehot"]
    binary_imputer = preprocess.named_transformers_["binary"]
    transformed_names = list(preprocess.get_feature_names_out())
    coefficients = model.coef_.ravel()
    if len(transformed_names) != 41 or len(coefficients) != 41 or len(model.intercept_) != 1:
        raise ValueError("Frozen transformed feature/intercept/coefficient count differs from expected 41/1/41")
    indicator_positions = np.asarray(numeric_imputer.indicator_.features_, dtype=int)
    indicator_features = [numeric_features[position] for position in indicator_positions]

    normalized = normalize_binary(data[features], binary_features)
    feature_rows = []
    preprocessing_rows = []
    name_to_coefficient = dict(zip(transformed_names, coefficients, strict=True))
    scaler_index = 0
    for index, feature in enumerate(numeric_features):
        transformed_name = f"numeric__{feature}"
        feature_rows.append({
            "record_type": "transformed_feature", "original_variable": feature, "transformed_feature": transformed_name,
            "feature_type": "numeric_value", "imputation": "median from locked development data",
            "missing_indicator": feature in indicator_features, "scaling": "StandardScaler: (imputed value - mean) / scale",
            "raw_zero_meaning": "not applicable", "raw_one_meaning": "not applicable", "reference_category": "not applicable",
            "coefficient": name_to_coefficient[transformed_name], "coefficient_is_nonzero": abs(name_to_coefficient[transformed_name]) > 1e-10,
            "transform_order": transformed_names.index(transformed_name),
        })
        indicator_name = f"numeric__missingindicator_{feature}"
        indicator_present = feature in indicator_features
        preprocessing_rows.append({
            "component_type": "numeric", "original_variable": feature, "transformed_feature": transformed_name,
            "development_imputation_median": numeric_imputer.statistics_[index], "scaler_mean": numeric_scaler.mean_[scaler_index],
            "scaler_scale": numeric_scaler.scale_[scaler_index], "missing_indicator_generated": indicator_present,
            "missing_indicator_transformed_feature": indicator_name if indicator_present else "",
            "missing_indicator_scaler_mean": numeric_scaler.mean_[len(numeric_features) + list(indicator_positions).index(index)] if indicator_present else np.nan,
            "missing_indicator_scaler_scale": numeric_scaler.scale_[len(numeric_features) + list(indicator_positions).index(index)] if indicator_present else np.nan,
            "missing_indicator_coefficient": name_to_coefficient[indicator_name] if indicator_present else np.nan,
            "imputation_or_encoding_rule": "SimpleImputer(strategy=median, add_indicator=True)",
        })
        scaler_index += 1
        if indicator_present:
            feature_rows.append({
                "record_type": "transformed_feature", "original_variable": feature, "transformed_feature": indicator_name,
                "feature_type": "numeric_missing_indicator", "imputation": "not applicable; 1 if original value missing, else 0",
                "missing_indicator": True, "scaling": "StandardScaler applied to 0/1 indicator",
                "raw_zero_meaning": "original value observed", "raw_one_meaning": "original value missing", "reference_category": "not applicable",
                "coefficient": name_to_coefficient[indicator_name], "coefficient_is_nonzero": abs(name_to_coefficient[indicator_name]) > 1e-10,
                "transform_order": transformed_names.index(indicator_name),
            })

    for category_index, feature in enumerate(categorical_features):
        for category in onehot.categories_[category_index]:
            transformed_name = f"categorical__{feature}_{category}"
            feature_rows.append({
                "record_type": "transformed_feature", "original_variable": feature, "transformed_feature": transformed_name,
                "feature_type": "one_hot_categorical", "imputation": "most frequent category from locked development data",
                "missing_indicator": False, "scaling": "not standardized", "raw_zero_meaning": f"gender is not {category}",
                "raw_one_meaning": f"gender is {category}", "reference_category": "none; OneHotEncoder(drop=None) retains all categories",
                "coefficient": name_to_coefficient[transformed_name], "coefficient_is_nonzero": abs(name_to_coefficient[transformed_name]) > 1e-10,
                "transform_order": transformed_names.index(transformed_name),
            })
        preprocessing_rows.append({
            "component_type": "categorical", "original_variable": feature, "transformed_feature": " | ".join(f"categorical__{feature}_{x}" for x in onehot.categories_[category_index]),
            "development_imputation_median": np.nan, "scaler_mean": np.nan, "scaler_scale": np.nan,
            "missing_indicator_generated": False, "missing_indicator_transformed_feature": "",
            "missing_indicator_scaler_mean": np.nan, "missing_indicator_scaler_scale": np.nan,
            "missing_indicator_coefficient": np.nan,
            "imputation_or_encoding_rule": f"SimpleImputer(most_frequent={categorical_imputer.statistics_[category_index]}); OneHotEncoder(drop=None, categories={list(onehot.categories_[category_index])})",
        })

    for index, feature in enumerate(binary_features):
        transformed_name = f"binary__{feature}"
        one_meaning = "yes/present before T0"
        zero_meaning = "no/absent before T0"
        if feature == "ventilation_no_record_pre_t0":
            one_meaning = "no qualifying ventilation record before T0"
            zero_meaning = "at least one qualifying ventilation record before T0"
        feature_rows.append({
            "record_type": "transformed_feature", "original_variable": feature, "transformed_feature": transformed_name,
            "feature_type": "binary", "imputation": "most frequent from locked development data",
            "missing_indicator": False, "scaling": "not standardized", "raw_zero_meaning": zero_meaning,
            "raw_one_meaning": one_meaning, "reference_category": "0",
            "coefficient": name_to_coefficient[transformed_name], "coefficient_is_nonzero": abs(name_to_coefficient[transformed_name]) > 1e-10,
            "transform_order": transformed_names.index(transformed_name),
        })
        preprocessing_rows.append({
            "component_type": "binary", "original_variable": feature, "transformed_feature": transformed_name,
            "development_imputation_median": np.nan, "scaler_mean": np.nan, "scaler_scale": np.nan,
            "missing_indicator_generated": False, "missing_indicator_transformed_feature": "",
            "missing_indicator_scaler_mean": np.nan, "missing_indicator_scaler_scale": np.nan,
            "missing_indicator_coefficient": np.nan,
            "imputation_or_encoding_rule": f"SimpleImputer(most_frequent={binary_imputer.statistics_[index]}); normalized 0/1; not standardized",
        })

    feature_mapping = pd.DataFrame(feature_rows).sort_values("transform_order").reset_index(drop=True)
    if feature_mapping["transformed_feature"].tolist() != transformed_names:
        raise ValueError("Exported feature order does not match frozen pipeline")
    preprocessing_parameters = pd.DataFrame(preprocessing_rows)
    model_specification = feature_mapping.copy()
    model_specification.insert(0, "model_type", "L1-penalized logistic regression")
    model_specification.insert(1, "final_C", model.C)
    model_specification.insert(2, "solver", model.solver)
    model_specification.insert(3, "l1_ratio", model.l1_ratio)
    model_specification.insert(4, "intercept", model.intercept_[0])
    model_specification.insert(5, "development_n", 23667)
    model_specification.insert(6, "development_events", 460)
    model_specification.insert(7, "original_variable_count", 27)
    model_specification.insert(8, "transformed_feature_count", 41)
    model_specification.insert(9, "final_nonzero_coefficient_count", int((np.abs(coefficients) > 1e-10).sum()))
    model_specification.insert(10, "random_seed", run_config["random_state"])
    model_specification.insert(11, "scikit_learn_version", sklearn.__version__)

    manual_matrix = manual_transformed_matrix(
        normalized, numeric_features, categorical_features, binary_features,
        numeric_imputer.statistics_, numeric_scaler.mean_, numeric_scaler.scale_, indicator_positions,
        categorical_imputer.statistics_, list(onehot.categories_), binary_imputer.statistics_,
    )
    frozen_matrix = preprocess.transform(normalized)
    frozen_matrix = frozen_matrix.toarray() if hasattr(frozen_matrix, "toarray") else np.asarray(frozen_matrix)
    transform_max_difference = float(np.max(np.abs(manual_matrix - frozen_matrix)))
    if transform_max_difference >= 1e-12:
        raise RuntimeError(f"Manual transformed matrix differs from frozen pipeline: {transform_max_difference}")
    manual_probability = expit(model.intercept_[0] + manual_matrix @ coefficients)
    pipeline_probability = pipeline.predict_proba(normalized)[:, 1]

    rng = np.random.default_rng(SEED)
    development_index = np.flatnonzero(data["temporal_split"].eq("development_2008_2016").to_numpy())
    temporal_index = np.flatnonzero(data["temporal_split"].eq("temporal_test_2017_2022").to_numpy())
    selected_index = np.concatenate([rng.choice(development_index, 100, replace=False), rng.choice(temporal_index, 100, replace=False)])
    selected_difference = np.abs(manual_probability[selected_index] - pipeline_probability[selected_index])
    if float(selected_difference.max()) >= 1e-10:
        raise RuntimeError("Independent reproduction does not meet max absolute difference <1e-10")

    numeric_missing_count = normalized[numeric_features].isna().sum(axis=1)
    no_missing = int(np.flatnonzero(numeric_missing_count.eq(0).to_numpy())[0])
    multiple_missing_candidates = np.flatnonzero(numeric_missing_count.ge(2).to_numpy())
    if not len(multiple_missing_candidates):
        raise RuntimeError("No example with multiple numeric missing values found")
    multiple_missing = int(multiple_missing_candidates[np.argmax(numeric_missing_count.iloc[multiple_missing_candidates].to_numpy())])
    high_risk_candidates = np.argsort(-manual_probability)
    high_risk = int(next(index for index in high_risk_candidates if index not in {no_missing, multiple_missing}))
    examples = []
    for label, index in [("no_missing", no_missing), ("multiple_missing", multiple_missing), ("high_risk", high_risk)]:
        original = {feature: (None if pd.isna(normalized.iloc[index][feature]) else normalized.iloc[index][feature]) for feature in features}
        transformed = {name: float(value) for name, value in zip(transformed_names, manual_matrix[index], strict=True)}
        examples.append({
            "example_type": label, "stay_id": int(data.iloc[index]["stay_id"]), "temporal_split": data.iloc[index]["temporal_split"],
            "numeric_missing_count": int(numeric_missing_count.iloc[index]), "original_inputs_json": json.dumps(original, ensure_ascii=False, default=float),
            "transformed_inputs_json": json.dumps(transformed, ensure_ascii=False),
            "linear_predictor_manual": float(model.intercept_[0] + manual_matrix[index] @ coefficients),
            "predicted_probability_manual": float(manual_probability[index]), "predicted_probability_frozen_pipeline": float(pipeline_probability[index]),
            "absolute_difference": float(abs(manual_probability[index] - pipeline_probability[index])),
        })
    examples = pd.DataFrame(examples)

    model_specification.to_csv(OUTPUT_DIR / "Table_S3_final_lasso_model_specification_v1.0.csv", index=False)
    preprocessing_parameters.to_csv(OUTPUT_DIR / "Table_S3_preprocessing_parameters_v1.0.csv", index=False)
    feature_mapping.to_csv(OUTPUT_DIR / "lasso_feature_mapping_v1.0.csv", index=False)
    examples.to_csv(OUTPUT_DIR / "lasso_reproduction_examples_v1.0.csv", index=False)
    outputs = [
        OUTPUT_DIR / "Table_S3_final_lasso_model_specification_v1.0.csv",
        OUTPUT_DIR / "Table_S3_preprocessing_parameters_v1.0.csv",
        OUTPUT_DIR / "lasso_feature_mapping_v1.0.csv",
        OUTPUT_DIR / "lasso_reproduction_examples_v1.0.csv",
    ]
    validation = {
        "status": "PASS",
        "analysis_type": "frozen_pipeline_parameter_export_and_independent_probability_reproduction",
        "input_hashes": {
            "locked_primary_data": file_hash(DATA_PATH), "feature_manifest": file_hash(FEATURE_MANIFEST_PATH),
            "run_config": file_hash(RUN_CONFIG_PATH), "frozen_lasso_pipeline": file_hash(MODEL_PATH),
        },
        "model_specification": {
            "model_type": "L1-penalized logistic regression", "final_C": model.C, "solver": model.solver,
            "l1_ratio": model.l1_ratio, "sklearn_penalty_attribute": model.penalty,
            "intercept": float(model.intercept_[0]), "development_n": 23667, "development_events": 460,
            "original_variable_count": 27, "transformed_feature_count": 41,
            "nonzero_coefficient_count_abs_gt_1e_10": int((np.abs(coefficients) > 1e-10).sum()),
            "random_seed": run_config["random_state"], "scikit_learn_version": sklearn.__version__,
        },
        "qc": {
            "feature_order_exact_match": True, "manual_matrix_max_abs_difference": transform_max_difference,
            "development_random_sample_n": 100, "temporal_validation_random_sample_n": 100,
            "random_sample_max_abs_probability_difference": float(selected_difference.max()),
            "random_sample_mean_abs_probability_difference": float(selected_difference.mean()),
            "random_sample_requirement_max_abs_lt_1e_10": True,
            "age_missing_indicator_generated": "age" in indicator_features,
            "sodium_last_missing_indicator_generated": "sodium_last" in indicator_features,
        },
        "reproduction_examples": examples[["example_type", "stay_id", "temporal_split", "numeric_missing_count", "absolute_difference"]].to_dict(orient="records"),
        "prohibited_actions": {
            "pipeline_refit": False, "imputer_refit": False, "scaler_refit": False, "encoder_refit": False,
            "lasso_refit": False, "coefficient_change": False, "feature_change": False, "cohort_change": False,
            "recalibration": False, "xgboost": False, "threshold_selection": False,
        },
        "output_hashes": {path.name: file_hash(path) for path in outputs},
    }
    (OUTPUT_DIR / "lasso_reproduction_validation_v1.0.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(model_specification[["transformed_feature", "coefficient", "coefficient_is_nonzero"]].to_string(index=False))
    print(examples[["example_type", "stay_id", "numeric_missing_count", "predicted_probability_manual", "absolute_difference"]].to_string(index=False))
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
