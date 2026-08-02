"""Train the prespecified 32-feature sodium-process extended models."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 20260729
NONZERO_THRESHOLD = 1e-10
EXTREME_PROBABILITY_THRESHOLD = 1e-6
LASSO_C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

NUMERIC_FEATURES = [
    "age",
    "sodium_first",
    "sodium_last",
    "sodium_min",
    "sodium_max",
    "sodium_delta",
    "sodium_n",
    "creatinine_last",
    "bun_last",
    "chloride_last",
    "potassium_last",
    "bicarbonate_last",
    "glucose_last",
    "heart_rate_last",
    "mbp_last",
    "resp_rate_last",
    "spo2_last",
    "temperature_last",
    "uo_mlkghr_6hr_clean",
    "uo_mlkghr_12hr_clean",
]
CATEGORICAL_FEATURES = ["gender"]
BINARY_FEATURES = [
    "invasive_vent_pre_t0",
    "noninvasive_vent_pre_t0",
    "hfnc_pre_t0",
    "supplemental_oxygen_pre_t0",
    "tracheostomy_pre_t0",
    "ventilation_no_record_pre_t0",
    "loop_diuretic_pre_t0",
    "vasoactive_pre_t0",
    "rrt_pre_t0",
    "iv_isotonic_crystalloid_pre_t0",
    "dextrose_or_hypotonic_fluid_pre_t0",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
ADDED_SODIUM_FEATURES = [
    "sodium_n",
    "sodium_first",
    "sodium_min",
    "sodium_max",
    "sodium_delta",
]


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def normalize_binary(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in BINARY_FEATURES:
        normalized[column] = (
            normalized[column].astype(str).str.lower().isin(["true", "t", "1"])
        ).astype(int)
    return normalized


def outcome(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "t", "1"]).astype(int)


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    binary = SimpleImputer(strategy="most_frequent")
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("binary", binary, BINARY_FEATURES),
        ]
    )


def make_pipeline(kind: str, c_value: float = 1.0) -> Pipeline:
    if kind == "logistic":
        classifier = LogisticRegression(
            C=np.inf,
            solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
    elif kind == "lasso":
        classifier = LogisticRegression(
            l1_ratio=1.0,
            solver="saga",
            C=c_value,
            max_iter=10000,
            tol=1e-4,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("preprocess", make_preprocessor()), ("model", classifier)])


def metric_row(
    model: str,
    dataset: str,
    y_true: pd.Series,
    probability: np.ndarray,
) -> dict:
    return {
        "model": model,
        "dataset": dataset,
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
        "event_rate_pct": round(float(y_true.mean() * 100), 3),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
    }


def write_predictions(
    output_dir: Path,
    model: str,
    dataset: str,
    ids: pd.Series,
    y_true: pd.Series,
    probability: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "stay_id": ids.to_numpy(),
            "outcome_na_ge_151": y_true.to_numpy(),
            "predicted_risk": probability,
        }
    ).to_csv(output_dir / f"{model}_{dataset}_predictions.csv", index=False)


def dense_matrix(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def coefficient_frame(
    pipeline: Pipeline,
    model: str,
    fit_scope: str,
    outer_fold: int | None = None,
    best_c: float | None = None,
) -> pd.DataFrame:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_.ravel()
    frame = pd.DataFrame(
        {
            "model": model,
            "fit_scope": fit_scope,
            "outer_fold": outer_fold,
            "best_C": best_c,
            "transformed_feature": names,
            "coefficient": coefficients,
        }
    )
    frame["abs_coefficient"] = frame["coefficient"].abs()
    frame["nonzero"] = frame["abs_coefficient"] > NONZERO_THRESHOLD
    return frame


def probability_audit(
    model: str,
    fit_scope: str,
    probability: np.ndarray,
) -> dict:
    probability = np.asarray(probability)
    return {
        "model": model,
        "fit_scope": fit_scope,
        "n": int(len(probability)),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "n_p_lt_1e_6": int((probability < EXTREME_PROBABILITY_THRESHOLD).sum()),
        "n_p_gt_1_minus_1e_6": int(
            (probability > 1 - EXTREME_PROBABILITY_THRESHOLD).sum()
        ),
        "n_p_lt_1e_4": int((probability < 1e-4).sum()),
        "n_p_gt_1_minus_1e_4": int((probability > 1 - 1e-4).sum()),
    }


def convergence_audit(
    pipeline: Pipeline,
    model: str,
    fit_scope: str,
    warning_messages: list[str],
) -> dict:
    classifier = pipeline.named_steps["model"]
    coefficients = classifier.coef_.ravel()
    max_iter = int(classifier.max_iter)
    n_iter = int(np.max(classifier.n_iter_))
    return {
        "model": model,
        "fit_scope": fit_scope,
        "n_iter": n_iter,
        "max_iter": max_iter,
        "converged_by_iteration_limit": bool(n_iter < max_iter),
        "convergence_warning": any(
            "converge" in message.lower() for message in warning_messages
        ),
        "warning_count": int(len(warning_messages)),
        "warning_messages": " | ".join(sorted(set(warning_messages))),
        "coefficient_count": int(len(coefficients)),
        "max_abs_coefficient": float(np.abs(coefficients).max()),
        "n_abs_coefficient_gt_5": int((np.abs(coefficients) > 5).sum()),
        "n_abs_coefficient_gt_10": int((np.abs(coefficients) > 10).sum()),
    }


def fit_with_warning_capture(pipeline: Pipeline, x, y) -> tuple[Pipeline, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(x, y)
    messages = [str(item.message) for item in caught]
    return pipeline, messages


def design_matrix_audit(
    pipeline: Pipeline,
    x: pd.DataFrame,
) -> dict:
    transformed = dense_matrix(pipeline.named_steps["preprocess"].transform(x))
    augmented = np.column_stack([np.ones(len(transformed)), transformed])
    rank = int(np.linalg.matrix_rank(augmented))
    columns = int(augmented.shape[1])
    return {
        "rows": int(augmented.shape[0]),
        "transformed_columns_without_intercept": int(transformed.shape[1]),
        "columns_with_intercept": columns,
        "rank_with_intercept": rank,
        "rank_deficiency": int(columns - rank),
        "full_column_rank": bool(rank == columns),
        "audit_note": (
            "Includes intercept; deterministic sodium_delta relation is expected "
            "to produce rank deficiency."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-role", required=True)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists and will not be overwritten: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    data = normalize_binary(pd.read_csv(input_path))
    missing = sorted(set(FEATURES).difference(data.columns))
    if missing:
        raise ValueError(f"Missing extended features: {missing}")
    if len(FEATURES) != 32 or len(set(FEATURES)) != 32:
        raise ValueError("Extended model must contain 32 unique variables")
    if data["stay_id"].duplicated().any():
        raise ValueError("Duplicated stay_id")
    delta_error = (
        data["sodium_delta"] - (data["sodium_last"] - data["sodium_first"])
    ).abs()
    if not bool((delta_error == 0).all()):
        raise ValueError("sodium_delta identity does not hold in all rows")

    development = data.loc[
        data["temporal_split"] == "development_2008_2016"
    ].copy()
    temporal_test = data.loc[
        data["temporal_split"] == "temporal_test_2017_2022"
    ].copy()
    if len(development) + len(temporal_test) != len(data):
        raise ValueError("Unexpected temporal_split value")

    x_dev = development[FEATURES]
    y_dev = outcome(development["outcome_na_ge_151"])
    x_test = temporal_test[FEATURES]
    y_test = outcome(temporal_test["outcome_na_ge_151"])
    outer_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    inner_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1
    )
    folds = list(outer_cv.split(x_dev, y_dev))
    fold_assignment = np.zeros(len(development), dtype=int)
    metrics = []
    convergence_rows = []
    probability_rows = []

    logistic_oof = np.zeros(len(development), dtype=float)
    for fold_number, (train_index, valid_index) in enumerate(folds, start=1):
        fold_assignment[valid_index] = fold_number
        fitted, warning_messages = fit_with_warning_capture(
            make_pipeline("logistic"),
            x_dev.iloc[train_index],
            y_dev.iloc[train_index],
        )
        fold_probability = fitted.predict_proba(x_dev.iloc[valid_index])[:, 1]
        logistic_oof[valid_index] = fold_probability
        convergence_rows.append(
            convergence_audit(
                fitted, "logistic", f"outer_fold_{fold_number}", warning_messages
            )
        )
        probability_rows.append(
            probability_audit(
                "logistic", f"outer_fold_{fold_number}_validation", fold_probability
            )
        )

    logistic, logistic_warnings = fit_with_warning_capture(
        make_pipeline("logistic"), x_dev, y_dev
    )
    logistic_test = logistic.predict_proba(x_test)[:, 1]
    logistic_dev_fitted = logistic.predict_proba(x_dev)[:, 1]
    joblib.dump(logistic, output_dir / "logistic_final_development.joblib")
    metrics.extend(
        [
            metric_row("logistic", "development_oof", y_dev, logistic_oof),
            metric_row("logistic", "temporal_test", y_test, logistic_test),
        ]
    )
    convergence_rows.append(
        convergence_audit(
            logistic, "logistic", "final_full_development", logistic_warnings
        )
    )
    probability_rows.extend(
        [
            probability_audit("logistic", "development_oof", logistic_oof),
            probability_audit(
                "logistic", "final_development_fitted", logistic_dev_fitted
            ),
            probability_audit("logistic", "temporal_test", logistic_test),
        ]
    )
    logistic_design_audit = design_matrix_audit(logistic, x_dev)
    coefficient_frame(
        logistic, "logistic", "final_full_development"
    ).to_csv(output_dir / "logistic_final_transformed_coefficients.csv", index=False)
    write_predictions(
        output_dir,
        "logistic",
        "development_oof",
        development["stay_id"],
        y_dev,
        logistic_oof,
    )
    write_predictions(
        output_dir,
        "logistic",
        "temporal_test",
        temporal_test["stay_id"],
        y_test,
        logistic_test,
    )

    lasso_oof = np.zeros(len(development), dtype=float)
    outer_c_values = []
    outer_coefficient_frames = []
    for fold_number, (train_index, valid_index) in enumerate(folds, start=1):
        search = GridSearchCV(
            make_pipeline("lasso"),
            {"model__C": LASSO_C_GRID},
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            search.fit(x_dev.iloc[train_index], y_dev.iloc[train_index])
        best = search.best_estimator_
        best_c = float(search.best_params_["model__C"])
        outer_c_values.append(best_c)
        fold_probability = best.predict_proba(x_dev.iloc[valid_index])[:, 1]
        lasso_oof[valid_index] = fold_probability
        outer_coefficient_frames.append(
            coefficient_frame(
                best,
                "lasso",
                "outer_training_fold",
                outer_fold=fold_number,
                best_c=best_c,
            )
        )
        convergence_rows.append(
            convergence_audit(
                best,
                "lasso",
                f"outer_fold_{fold_number}",
                [str(item.message) for item in caught],
            )
        )
        probability_rows.append(
            probability_audit(
                "lasso", f"outer_fold_{fold_number}_validation", fold_probability
            )
        )

    final_search = GridSearchCV(
        make_pipeline("lasso"),
        {"model__C": LASSO_C_GRID},
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        final_search.fit(x_dev, y_dev)
    lasso = final_search.best_estimator_
    final_c = float(final_search.best_params_["model__C"])
    lasso_test = lasso.predict_proba(x_test)[:, 1]
    lasso_dev_fitted = lasso.predict_proba(x_dev)[:, 1]
    joblib.dump(lasso, output_dir / "lasso_final_development.joblib")
    metrics.extend(
        [
            metric_row("lasso", "development_nested_oof", y_dev, lasso_oof),
            metric_row("lasso", "temporal_test", y_test, lasso_test),
        ]
    )
    convergence_rows.append(
        convergence_audit(
            lasso,
            "lasso",
            "final_full_development",
            [str(item.message) for item in caught],
        )
    )
    probability_rows.extend(
        [
            probability_audit("lasso", "development_nested_oof", lasso_oof),
            probability_audit(
                "lasso", "final_development_fitted", lasso_dev_fitted
            ),
            probability_audit("lasso", "temporal_test", lasso_test),
        ]
    )

    outer_coefficients = pd.concat(outer_coefficient_frames, ignore_index=True)
    outer_coefficients.to_csv(
        output_dir / "lasso_outer_fold_transformed_coefficients.csv", index=False
    )
    final_coefficients = coefficient_frame(
        lasso,
        "lasso",
        "final_full_development",
        best_c=final_c,
    )
    final_coefficients.to_csv(
        output_dir / "lasso_final_transformed_coefficients.csv", index=False
    )

    added_names = [f"numeric__{feature}" for feature in ADDED_SODIUM_FEATURES]
    added_outer = outer_coefficients.loc[
        outer_coefficients["transformed_feature"].isin(added_names)
    ].copy()
    added_outer["original_feature"] = added_outer[
        "transformed_feature"
    ].str.removeprefix("numeric__")
    added_outer.to_csv(
        output_dir / "added_sodium_features_outer_fold_audit.csv", index=False
    )
    selection_summary = (
        added_outer.groupby("original_feature", as_index=False)
        .agg(
            folds_evaluated=("outer_fold", "nunique"),
            folds_nonzero=("nonzero", "sum"),
            selection_frequency=("nonzero", "mean"),
            coefficient_min=("coefficient", "min"),
            coefficient_max=("coefficient", "max"),
            coefficient_mean=("coefficient", "mean"),
        )
        .sort_values("original_feature")
    )
    group_stability = (
        added_outer.groupby("outer_fold")["nonzero"].any().rename("any_added_nonzero")
    )
    selection_summary.to_csv(
        output_dir / "added_sodium_features_selection_summary.csv", index=False
    )
    pd.DataFrame(
        {
            "group_definition": ["at_least_one_of_five_added_features_nonzero"],
            "folds_evaluated": [int(group_stability.size)],
            "folds_selected": [int(group_stability.sum())],
            "selection_frequency": [float(group_stability.mean())],
        }
    ).to_csv(
        output_dir / "added_sodium_feature_group_stability.csv", index=False
    )

    write_predictions(
        output_dir,
        "lasso",
        "development_nested_oof",
        development["stay_id"],
        y_dev,
        lasso_oof,
    )
    write_predictions(
        output_dir,
        "lasso",
        "temporal_test",
        temporal_test["stay_id"],
        y_test,
        lasso_test,
    )

    pd.DataFrame(metrics).to_csv(
        output_dir / "performance_metrics.csv", index=False
    )
    pd.DataFrame(convergence_rows).to_csv(
        output_dir / "convergence_coefficient_audit.csv", index=False
    )
    pd.DataFrame(probability_rows).to_csv(
        output_dir / "extreme_probability_audit.csv", index=False
    )
    (output_dir / "logistic_design_matrix_rank_audit.json").write_text(
        json.dumps(logistic_design_audit, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "stay_id": development["stay_id"].to_numpy(),
            "outer_fold": fold_assignment,
        }
    ).to_csv(output_dir / "development_outer_fold_assignments.csv", index=False)
    pd.DataFrame(
        [
            {
                "feature_order": index,
                "feature": feature,
                "feature_type": (
                    "numeric"
                    if feature in NUMERIC_FEATURES
                    else "categorical"
                    if feature in CATEGORICAL_FEATURES
                    else "binary"
                ),
                "added_sodium_process_feature": feature in ADDED_SODIUM_FEATURES,
            }
            for index, feature in enumerate(FEATURES, start=1)
        ]
    ).to_csv(output_dir / "extended_feature_manifest.csv", index=False)

    config = {
        "input_file": str(input_path),
        "input_sha256": file_hash(input_path),
        "output_directory": str(output_dir),
        "analysis_role": args.analysis_role,
        "analysis_status": "prespecified_secondary_extended_analysis",
        "core_model_remains_locked": True,
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "added_sodium_features": ADDED_SODIUM_FEATURES,
        "sodium_delta_identity_confirmed": True,
        "logistic_coefficient_interpretation_prohibited": True,
        "lasso_single_added_feature_clinical_ranking_prohibited": True,
        "development_split": "2008-2016",
        "temporal_test_split": "2017-2022",
        "development_n": int(len(development)),
        "development_events": int(y_dev.sum()),
        "temporal_test_n": int(len(temporal_test)),
        "temporal_test_events": int(y_test.sum()),
        "outer_folds": 5,
        "inner_folds_lasso": 5,
        "random_state": RANDOM_STATE,
        "lasso_C_grid": LASSO_C_GRID,
        "lasso_final_C": final_c,
        "lasso_outer_C_values": outer_c_values,
        "nonzero_threshold": NONZERO_THRESHOLD,
        "bootstrap_runs_later_comparison": 1000,
        "class_weight": None,
        "missing_value_processing": (
            "training-fold-only median imputation plus missing indicators"
        ),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"lasso_final_C={final_c}")
    print(f"lasso_outer_C_values={outer_c_values}")
    print(f"logistic_design_matrix_audit={logistic_design_audit}")


if __name__ == "__main__":
    main()
