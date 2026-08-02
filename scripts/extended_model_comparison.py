"""Compare locked 27-feature and extended 32-feature model predictions."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT = Path(__file__).resolve().parents[1]
MODELING = PROJECT / "modeling"
OUTPUT = MODELING / "v1.1_sodium_process_extended"
BOOTSTRAP_RUNS = 1000
RANDOM_SEED = 20260731

COHORTS = {
    "primary": {
        "core": MODELING / "v1.0_locked_primary",
        "extended": OUTPUT / "primary",
        "dataset": PROJECT / "data" / "primary_cohort_model_dataset_v1.0.csv",
    },
    "high_certainty_sensitivity": {
        "core": MODELING / "v1.0_high_certainty_sensitivity",
        "extended": OUTPUT / "high_certainty_sensitivity",
        "dataset": (
            PROJECT
            / "data"
            / "high_certainty_sensitivity_model_dataset_v1.0.csv"
        ),
    },
}


def model_dataset_name(model: str) -> str:
    return "development_oof" if model == "logistic" else "development_nested_oof"


def metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
    }


def read_test_predictions(directory: Path, model: str, label: str) -> pd.DataFrame:
    path = directory / f"{model}_temporal_test_predictions.csv"
    frame = pd.read_csv(path)
    required = {"stay_id", "outcome_na_ge_151", "predicted_risk"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: missing required prediction columns")
    if frame["stay_id"].duplicated().any():
        raise ValueError(f"{path}: duplicated stay_id")
    if not np.isfinite(frame["predicted_risk"]).all():
        raise ValueError(f"{path}: non-finite prediction")
    return frame.rename(
        columns={
            "outcome_na_ge_151": f"outcome_{label}",
            "predicted_risk": f"prediction_{label}",
        }
    )


def aligned_predictions(spec: dict, model: str) -> tuple[pd.DataFrame, dict]:
    core = read_test_predictions(spec["core"], model, "core")
    extended = read_test_predictions(spec["extended"], model, "extended")
    if set(core["stay_id"]) != set(extended["stay_id"]):
        raise ValueError(f"{model}: core/extended stay_id sets differ")
    merged = core.merge(
        extended, on="stay_id", how="inner", validate="one_to_one"
    ).sort_values("stay_id")
    if not np.array_equal(
        merged["outcome_core"].to_numpy(),
        merged["outcome_extended"].to_numpy(),
    ):
        raise ValueError(f"{model}: core/extended outcomes differ")
    probabilities = merged[["prediction_core", "prediction_extended"]].to_numpy()
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{model}: probability outside [0, 1]")
    audit = {
        "model": model,
        "core_n": int(len(core)),
        "extended_n": int(len(extended)),
        "aligned_n": int(len(merged)),
        "stay_id_set_match": True,
        "outcome_match": True,
        "predictions_complete": True,
    }
    return merged, audit


def bootstrap_indices(y: np.ndarray, n_runs: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = []
    while len(indices) < n_runs:
        sample = rng.integers(0, len(y), len(y))
        if np.unique(y[sample]).size == 2:
            indices.append(sample)
    return indices


def paired_increment_rows(
    cohort: str,
    model: str,
    merged: pd.DataFrame,
    samples: list[np.ndarray],
) -> list[dict]:
    y = merged["outcome_core"].to_numpy()
    core = merged["prediction_core"].to_numpy()
    extended = merged["prediction_extended"].to_numpy()
    core_metrics = metric_values(y, core)
    extended_metrics = metric_values(y, extended)
    estimates = {
        "delta_auc": extended_metrics["roc_auc"] - core_metrics["roc_auc"],
        "delta_ap": (
            extended_metrics["average_precision"]
            - core_metrics["average_precision"]
        ),
        "delta_brier_improvement": (
            core_metrics["brier_score"] - extended_metrics["brier_score"]
        ),
    }
    draws = {metric: [] for metric in estimates}
    for sample in samples:
        sampled_y = y[sample]
        core_sample = metric_values(sampled_y, core[sample])
        extended_sample = metric_values(sampled_y, extended[sample])
        draws["delta_auc"].append(
            extended_sample["roc_auc"] - core_sample["roc_auc"]
        )
        draws["delta_ap"].append(
            extended_sample["average_precision"]
            - core_sample["average_precision"]
        )
        draws["delta_brier_improvement"].append(
            core_sample["brier_score"] - extended_sample["brier_score"]
        )
    rows = []
    for metric, estimate in estimates.items():
        low, high = np.quantile(draws[metric], [0.025, 0.975])
        rows.append(
            {
                "cohort": cohort,
                "model": model,
                "metric": metric,
                "estimate": float(estimate),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "positive_means_extended_improvement": True,
                "bootstrap_type": "paired_patient_level",
                "bootstrap_runs": len(samples),
            }
        )
    return rows


def absolute_performance_rows(cohort: str, spec: dict) -> list[dict]:
    rows = []
    for feature_set, directory in [
        ("core_27", spec["core"]),
        ("extended_32", spec["extended"]),
    ]:
        source = (
            directory / "audit" / "performance_calibration_bootstrap_ci.csv"
        )
        metrics = pd.read_csv(source)
        test = metrics.loc[metrics["dataset"] == "temporal_test"].copy()
        for _, row in test.iterrows():
            rows.append(
                {
                    "cohort": cohort,
                    "feature_set": feature_set,
                    "model": row["model"],
                    "metric": row["metric"],
                    "estimate": float(row["estimate"]),
                    "ci95_low": float(row["ci95_low"]),
                    "ci95_high": float(row["ci95_high"]),
                    "bootstrap_runs": BOOTSTRAP_RUNS,
                }
            )
    return rows


def plot_calibration_comparison(cohort: str, spec: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for axis, model in zip(axes, ["logistic", "lasso"]):
        core = pd.read_csv(
            spec["core"]
            / "audit"
            / f"{model}_temporal_test_calibration_deciles.csv"
        )
        extended = pd.read_csv(
            spec["extended"]
            / "audit"
            / f"{model}_temporal_test_calibration_deciles.csv"
        )
        axis_max = max(
            0.02,
            float(
                max(
                    core["mean_predicted_risk"].max(),
                    core["observed_rate"].max(),
                    extended["mean_predicted_risk"].max(),
                    extended["observed_rate"].max(),
                )
                * 1.15
            ),
        )
        axis.plot(
            [0, axis_max],
            [0, axis_max],
            "--",
            color="0.5",
            label="ideal",
        )
        axis.plot(
            core["mean_predicted_risk"],
            core["observed_rate"],
            "o-",
            label="27-feature core",
        )
        axis.plot(
            extended["mean_predicted_risk"],
            extended["observed_rate"],
            "s-",
            label="32-feature extended",
        )
        axis.set_title(f"{model.title()}: temporal test", fontsize=11)
        axis.set_xlabel("Mean predicted risk", fontsize=9)
        axis.set_ylabel("Observed event rate", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.set_xlim(0, axis_max)
        axis.set_ylim(0, axis_max)
        axis.legend(frameon=False, fontsize=8)
    path = OUTPUT / f"{cohort}_calibration_27_vs_32.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    output_files = [
        OUTPUT / "absolute_performance_27_vs_32.csv",
        OUTPUT / "paired_bootstrap_increment.csv",
        OUTPUT / "prediction_alignment_audit.csv",
        OUTPUT / "comparison_run_config.json",
    ]
    if any(path.exists() for path in output_files):
        raise FileExistsError("Comparison output already exists and will not be overwritten")

    alignment_rows = []
    increment_rows = []
    absolute_rows = []
    cohort_audit = []
    plot_paths = []

    for cohort_index, (cohort, spec) in enumerate(COHORTS.items()):
        source = pd.read_csv(
            spec["dataset"],
            usecols=["subject_id", "stay_id", "temporal_split"],
        )
        test_source = source.loc[
            source["temporal_split"] == "temporal_test_2017_2022"
        ]
        subject_unique = test_source["subject_id"].nunique() == len(test_source)
        stay_unique = test_source["stay_id"].nunique() == len(test_source)
        if not subject_unique or not stay_unique:
            raise ValueError(f"{cohort}: test observations are not patient-unique")
        cohort_audit.append(
            {
                "cohort": cohort,
                "temporal_test_rows": int(len(test_source)),
                "unique_subjects": int(test_source["subject_id"].nunique()),
                "unique_stays": int(test_source["stay_id"].nunique()),
                "stay_sampling_equals_patient_sampling": True,
            }
        )

        aligned_by_model = {}
        for model in ["logistic", "lasso"]:
            merged, audit = aligned_predictions(spec, model)
            audit["cohort"] = cohort
            alignment_rows.append(audit)
            aligned_by_model[model] = merged

        reference_y = aligned_by_model["logistic"]["outcome_core"].to_numpy()
        if not np.array_equal(
            reference_y,
            aligned_by_model["lasso"]["outcome_core"].to_numpy(),
        ):
            raise ValueError(f"{cohort}: logistic/lasso aligned outcomes differ")
        samples = bootstrap_indices(
            reference_y,
            BOOTSTRAP_RUNS,
            RANDOM_SEED + cohort_index,
        )
        for model, merged in aligned_by_model.items():
            increment_rows.extend(
                paired_increment_rows(cohort, model, merged, samples)
            )
        absolute_rows.extend(absolute_performance_rows(cohort, spec))
        plot_paths.append(str(plot_calibration_comparison(cohort, spec)))

    pd.DataFrame(absolute_rows).to_csv(output_files[0], index=False)
    pd.DataFrame(increment_rows).to_csv(output_files[1], index=False)
    pd.DataFrame(alignment_rows).to_csv(output_files[2], index=False)
    config = {
        "analysis_status": "prespecified_secondary_extended_analysis",
        "core_model_remains_locked": True,
        "bootstrap_runs": BOOTSTRAP_RUNS,
        "bootstrap_seed": RANDOM_SEED,
        "paired_resampling": True,
        "same_indices_for_core_and_extended_within_each_resample": True,
        "delta_definitions": {
            "delta_auc": "AUC32 - AUC27",
            "delta_ap": "AP32 - AP27",
            "delta_brier_improvement": "Brier27 - Brier32",
        },
        "positive_delta_means_extended_improvement": True,
        "cohort_patient_uniqueness_audit": cohort_audit,
        "calibration_plots": plot_paths,
        "interpretation_boundary": (
            "Increment represents combined sodium trajectory and monitoring "
            "process information; individual added-feature clinical ranking "
            "is prohibited."
        ),
    }
    output_files[3].write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(increment_rows).to_string(index=False))
    print(pd.DataFrame(alignment_rows).to_string(index=False))


if __name__ == "__main__":
    main()
