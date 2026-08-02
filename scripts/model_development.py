"""Train locked 27-feature logistic and LASSO models without leakage."""

from pathlib import Path
import argparse
from hashlib import sha256
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT / "data" / "strict_primary_model_dataset_v0.2.csv"
DEFAULT_OUTPUT_DIR = PROJECT / "modeling" / "v0.1_exploratory"
RANDOM_STATE = 20260729

NUMERIC_FEATURES = [
    "age", "sodium_last", "creatinine_last", "bun_last", "chloride_last",
    "potassium_last", "bicarbonate_last", "glucose_last", "heart_rate_last",
    "mbp_last", "resp_rate_last", "spo2_last", "temperature_last",
    "uo_mlkghr_6hr_clean", "uo_mlkghr_12hr_clean",
]
CATEGORICAL_FEATURES = ["gender"]
BINARY_FEATURES = [
    "invasive_vent_pre_t0", "noninvasive_vent_pre_t0", "hfnc_pre_t0",
    "supplemental_oxygen_pre_t0", "tracheostomy_pre_t0",
    "ventilation_no_record_pre_t0", "loop_diuretic_pre_t0", "vasoactive_pre_t0",
    "rrt_pre_t0", "iv_isotonic_crystalloid_pre_t0",
    "dextrose_or_hypotonic_fluid_pre_t0",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
LASSO_C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
EXCLUDED_SODIUM_PROCESS_FEATURES = [
    "sodium_n", "sodium_first", "sodium_min", "sodium_max", "sodium_delta"
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


def make_pipeline(kind: str, c_value: float | None = None) -> Pipeline:
    if kind == "logistic":
        classifier = LogisticRegression(
            C=np.inf, solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE
        )
    elif kind == "lasso":
        classifier = LogisticRegression(
            l1_ratio=1.0, solver="saga", C=c_value or 1.0, max_iter=10000,
            tol=1e-4, random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("preprocess", make_preprocessor()), ("model", classifier)])


def metric_row(name: str, dataset: str, y_true: pd.Series, probability: np.ndarray) -> dict:
    return {
        "model": name,
        "dataset": dataset,
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
        "event_rate_pct": round(float(y_true.mean() * 100), 3),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 4),
        "average_precision": round(float(average_precision_score(y_true, probability)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probability)), 5),
    }


def write_predictions(
    output_dir: Path,
    name: str,
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
    ).to_csv(output_dir / f"{name}_{dataset}_predictions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-role", required=True)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists and will not be overwritten: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    data = normalize_binary(pd.read_csv(input_path))
    missing_features = sorted(set(FEATURES).difference(data.columns))
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")
    if len(FEATURES) != 27 or len(set(FEATURES)) != 27:
        raise ValueError("The locked core feature set must contain 27 unique variables")
    overlap = sorted(set(FEATURES).intersection(EXCLUDED_SODIUM_PROCESS_FEATURES))
    if overlap:
        raise ValueError(f"Excluded sodium process features entered model: {overlap}")
    if data["stay_id"].duplicated().any():
        raise ValueError("Duplicated stay_id in model dataset")
    development = data.loc[data["temporal_split"] == "development_2008_2016"].copy()
    temporal_test = data.loc[data["temporal_split"] == "temporal_test_2017_2022"].copy()
    if len(development) + len(temporal_test) != len(data):
        raise ValueError("Unexpected temporal_split value")
    x_dev, y_dev = development[FEATURES], outcome(development["outcome_na_ge_151"])
    x_test, y_test = temporal_test[FEATURES], outcome(temporal_test["outcome_na_ge_151"])

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
    rows = []

    logistic = make_pipeline("logistic")
    logistic_oof = cross_val_predict(
        logistic, x_dev, y_dev, cv=outer_cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    logistic.fit(x_dev, y_dev)
    logistic_test = logistic.predict_proba(x_test)[:, 1]
    joblib.dump(logistic, output_dir / "logistic_final_development.joblib")
    rows.extend(
        [
            metric_row("logistic", "development_oof", y_dev, logistic_oof),
            metric_row("logistic", "temporal_test", y_test, logistic_test),
        ]
    )
    write_predictions(
        output_dir, "logistic", "development_oof",
        development["stay_id"], y_dev, logistic_oof
    )
    write_predictions(
        output_dir, "logistic", "temporal_test",
        temporal_test["stay_id"], y_test, logistic_test
    )

    lasso_oof = np.zeros(len(development), dtype=float)
    outer_c_values = []
    for train_index, valid_index in outer_cv.split(x_dev, y_dev):
        search = GridSearchCV(
            make_pipeline("lasso"),
            {"model__C": LASSO_C_GRID},
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_dev.iloc[train_index], y_dev.iloc[train_index])
        lasso_oof[valid_index] = search.predict_proba(x_dev.iloc[valid_index])[:, 1]
        outer_c_values.append(float(search.best_params_["model__C"]))

    final_search = GridSearchCV(
        make_pipeline("lasso"),
        {"model__C": LASSO_C_GRID},
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )
    final_search.fit(x_dev, y_dev)
    lasso = final_search.best_estimator_
    lasso_test = lasso.predict_proba(x_test)[:, 1]
    joblib.dump(lasso, output_dir / "lasso_final_development.joblib")
    rows.extend(
        [
            metric_row("lasso", "development_nested_oof", y_dev, lasso_oof),
            metric_row("lasso", "temporal_test", y_test, lasso_test),
        ]
    )
    write_predictions(
        output_dir, "lasso", "development_nested_oof",
        development["stay_id"], y_dev, lasso_oof
    )
    write_predictions(
        output_dir, "lasso", "temporal_test",
        temporal_test["stay_id"], y_test, lasso_test
    )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "performance_metrics.csv", index=False)
    feature_manifest = pd.DataFrame(
        [
            {
                "feature_order": index,
                "feature": feature,
                "feature_type": (
                    "numeric" if feature in NUMERIC_FEATURES
                    else "categorical" if feature in CATEGORICAL_FEATURES
                    else "binary"
                ),
                "time_boundary": "T0_pre",
                "core_model": True,
            }
            for index, feature in enumerate(FEATURES, start=1)
        ]
    )
    feature_manifest.to_csv(output_dir / "core_feature_manifest.csv", index=False)
    config = {
        "input_file": str(input_path),
        "input_sha256": file_hash(input_path),
        "output_directory": str(output_dir),
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "excluded_sodium_process_features": EXCLUDED_SODIUM_PROCESS_FEATURES,
        "analysis_role": args.analysis_role,
        "protocol_amendment_version": "v1.0_2026-07-30",
        "result_boundary": "locked model run; researcher retains final clinical interpretation",
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
        "lasso_final_C": float(final_search.best_params_["model__C"]),
        "lasso_outer_C_values": outer_c_values,
        "class_weight": None,
        "missing_value_processing": "training-fold-only median imputation plus missing indicators",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(f"lasso_final_C={final_search.best_params_['model__C']}")
    print(f"lasso_outer_C_values={outer_c_values}")


if __name__ == "__main__":
    main()
