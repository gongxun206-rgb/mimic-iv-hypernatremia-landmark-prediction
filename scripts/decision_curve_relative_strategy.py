"""Paired bootstrap audit of frozen LASSO DCA versus fixed strategies."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT / "outputs" / "primary_model"
DCA_DIR = PROJECT / "outputs" / "dca_raw_probability"
DATA_PATH = PROJECT / "data" / "primary_model_dataset.csv"
OUTPUT_DIR = PROJECT / "outputs" / "dca_relative_strategy"
THRESHOLDS = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15])
BOOTSTRAP_RUNS = 1000
BOOTSTRAP_SEED = 20260731


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def net_benefits(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    n = len(y)
    flagged = probability >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    tn = int(np.sum(~flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    prevalence = float(y.mean())
    odds = threshold / (1 - threshold)
    model = tp / n - fp / n * odds
    treat_all = prevalence - (1 - prevalence) * odds
    treat_none = 0.0
    return {
        "n": n,
        "events": int(y.sum()),
        "flagged_n": int(flagged.sum()),
        "flagged_proportion": float(flagged.mean()),
        "captured_events": tp,
        "captured_event_proportion": tp / y.sum() if y.sum() else np.nan,
        "sensitivity": tp / y.sum() if y.sum() else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "nb_model": model,
        "nb_treat_all": treat_all,
        "nb_treat_none": treat_none,
        "delta_nb_vs_all": model - treat_all,
        "delta_nb_vs_none": model - treat_none,
        "net_reduction_interventions_per_100_vs_all": (model - treat_all) / odds * 100,
    }


def bootstrap_relative_strategy(y: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for iteration in range(1, BOOTSTRAP_RUNS + 1):
        index = rng.integers(0, len(y), len(y))
        sampled_y = y[index]
        sampled_probability = probability[index]
        for threshold in THRESHOLDS:
            metrics = net_benefits(sampled_y, sampled_probability, float(threshold))
            rows.append(
                {
                    "bootstrap_iteration": iteration,
                    "threshold_probability": float(threshold),
                    "n": metrics["n"],
                    "events": metrics["events"],
                    "nb_model": metrics["nb_model"],
                    "nb_treat_all": metrics["nb_treat_all"],
                    "nb_treat_none": metrics["nb_treat_none"],
                    "delta_nb_vs_all": metrics["delta_nb_vs_all"],
                    "delta_nb_vs_none": metrics["delta_nb_vs_none"],
                    "net_reduction_interventions_per_100_vs_all": metrics[
                        "net_reduction_interventions_per_100_vs_all"
                    ],
                }
            )
    return pd.DataFrame(rows)


def add_ci(point: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in point.iterrows():
        threshold = row["threshold_probability"]
        group = draws.loc[draws["threshold_probability"] == threshold]
        output = row.to_dict()
        for metric in [
            "delta_nb_vs_all",
            "delta_nb_vs_none",
            "net_reduction_interventions_per_100_vs_all",
        ]:
            output[f"{metric}_ci95_low_bootstrap"] = float(group[metric].quantile(0.025))
            output[f"{metric}_ci95_high_bootstrap"] = float(group[metric].quantile(0.975))
            output[f"{metric}_bootstrap_proportion_gt_0"] = float((group[metric] > 0).mean())
        rows.append(output)
    return pd.DataFrame(rows)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    prediction_path = MODEL_DIR / "lasso_temporal_test_predictions.csv"
    metrics_path = DCA_DIR / "dca_threshold_metrics.csv"
    prediction = pd.read_csv(prediction_path).sort_values("stay_id", kind="mergesort")
    if prediction["stay_id"].duplicated().any() or len(prediction) != 11246:
        raise ValueError("Temporal prediction stay_id integrity failed")
    y = prediction["outcome_na_ge_151"].to_numpy(dtype=int)
    probability = prediction["predicted_risk"].to_numpy(dtype=float)
    if int(y.sum()) != 363 or not np.isfinite(probability).all() or not np.all((probability >= 0) & (probability <= 1)):
        raise ValueError("Temporal prediction outcome/probability integrity failed")

    locked_data = pd.read_csv(DATA_PATH, usecols=["stay_id", "subject_id", "temporal_split"])
    temporal = locked_data.loc[locked_data["temporal_split"] == "temporal_test_2017_2022"]
    if temporal["stay_id"].nunique() != 11246 or temporal["subject_id"].nunique() != 11246:
        raise ValueError("Temporal validation is not one stay per subject")
    if set(temporal["stay_id"]) != set(prediction["stay_id"]):
        raise ValueError("Temporal prediction stays differ from locked cohort")

    reused_metrics = pd.read_csv(metrics_path)
    reused_metrics = reused_metrics.loc[
        (reused_metrics["dataset"] == "temporal_validation")
        & (reused_metrics["threshold_probability"].isin(THRESHOLDS))
    ].sort_values("threshold_probability")
    point = pd.DataFrame(
        [
            {"threshold_probability": float(threshold), **net_benefits(y, probability, float(threshold))}
            for threshold in THRESHOLDS
        ]
    )
    comparison_columns = [
        "flagged_n", "flagged_proportion", "captured_events", "captured_event_proportion",
        "sensitivity", "specificity", "ppv", "npv",
    ]
    check = point.merge(
        reused_metrics[["threshold_probability", *comparison_columns]],
        on="threshold_probability", suffixes=("_new", "_prior"), validate="one_to_one"
    )
    for column in comparison_columns:
        if not np.allclose(check[f"{column}_new"], check[f"{column}_prior"], rtol=0, atol=1e-12, equal_nan=True):
            raise RuntimeError(f"DCA point estimate mismatch for {column}")
    prior_nb = reused_metrics.rename(
        columns={
            "net_benefit_model": "nb_model",
            "net_benefit_treat_all": "nb_treat_all",
            "net_benefit_treat_none": "nb_treat_none",
        }
    )
    nb_check = point.merge(
        prior_nb[["threshold_probability", "nb_model", "nb_treat_all", "nb_treat_none"]],
        on="threshold_probability", suffixes=("_new", "_prior"), validate="one_to_one"
    )
    for column in ["nb_model", "nb_treat_all", "nb_treat_none"]:
        if not np.allclose(nb_check[f"{column}_new"], nb_check[f"{column}_prior"], rtol=0, atol=1e-12):
            raise RuntimeError(f"DCA point estimate mismatch for {column}")

    draws = bootstrap_relative_strategy(y, probability)
    increment = add_ci(point, draws)
    decision_columns = [
        "threshold_probability", "flagged_proportion", "captured_event_proportion", "sensitivity",
        "specificity", "ppv", "npv", "nb_model", "delta_nb_vs_all",
        "delta_nb_vs_all_ci95_low_bootstrap", "delta_nb_vs_all_ci95_high_bootstrap",
        "delta_nb_vs_none", "delta_nb_vs_none_ci95_low_bootstrap",
        "delta_nb_vs_none_ci95_high_bootstrap", "net_reduction_interventions_per_100_vs_all",
        "net_reduction_interventions_per_100_vs_all_ci95_low_bootstrap",
        "net_reduction_interventions_per_100_vs_all_ci95_high_bootstrap",
    ]
    increment.to_csv(OUTPUT_DIR / "dca_relative_strategy_increment.csv", index=False)
    increment[decision_columns].to_csv(OUTPUT_DIR / "dca_threshold_decision_table.csv", index=False)
    draws.to_csv(OUTPUT_DIR / "dca_relative_strategy_bootstrap.csv", index=False)

    output_paths = [
        OUTPUT_DIR / "dca_relative_strategy_increment.csv",
        OUTPUT_DIR / "dca_threshold_decision_table.csv",
        OUTPUT_DIR / "dca_relative_strategy_bootstrap.csv",
    ]
    validation = {
        "status": "PASS",
        "analysis_type": "paired_bootstrap_relative_dca_strategy_audit_only",
        "model": "locked_27_feature_primary_lasso",
        "probability_scale": "frozen_raw_probability_no_recalibration",
        "input_hashes": {
            "temporal_prediction": file_hash(prediction_path),
            "existing_dca_threshold_metrics": file_hash(metrics_path),
            "locked_primary_dataset": file_hash(DATA_PATH),
        },
        "data_integrity": {
            "n": int(len(prediction)), "events": int(y.sum()), "unique_stay_id": int(prediction["stay_id"].nunique()),
            "unique_subject_id": int(temporal["subject_id"].nunique()),
        },
        "bootstrap": {
            "runs": BOOTSTRAP_RUNS, "seed": BOOTSTRAP_SEED,
            "unit": "stay_id; verified one unique subject_id per temporal-validation stay",
            "paired_strategy_calculation": True,
        },
        "prespecified_thresholds": THRESHOLDS.tolist(),
        "existing_dca_point_estimates_exact_match": True,
        "output_hashes": {path.name: file_hash(path) for path in output_paths},
        "prohibited_actions": {
            "retrain": False, "recalibration": False, "feature_selection": False,
            "cohort_change": False, "xgboost": False, "clinical_threshold_selection": False,
        },
    }
    (OUTPUT_DIR / "dca_relative_strategy_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(increment.to_string(index=False))
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
