"""Decision-curve audit for frozen raw predictions from the 27-feature LASSO."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT / "outputs" / "primary_model"
DATA_PATH = PROJECT / "data" / "primary_model_dataset.csv"
OUTPUT_DIR = PROJECT / "outputs" / "dca_raw_probability"
GRID = np.round(np.arange(0.01, 0.1501, 0.001), 4)
REPRESENTATIVE_THRESHOLDS = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15])
BOOTSTRAP_RUNS = 1000
BOOTSTRAP_SEED = 20260731


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_predictions(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"stay_id", "outcome_na_ge_151", "predicted_risk"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{label}: expected columns {required}")
    if frame["stay_id"].duplicated().any():
        raise ValueError(f"{label}: duplicate stay_id")
    if not set(frame["outcome_na_ge_151"].unique()).issubset({0, 1}):
        raise ValueError(f"{label}: outcome is not binary")
    if not np.isfinite(frame["predicted_risk"]).all():
        raise ValueError(f"{label}: non-finite predicted risk")
    if not frame["predicted_risk"].between(0, 1).all():
        raise ValueError(f"{label}: predicted risk outside [0, 1]")
    return frame.sort_values("stay_id", kind="mergesort").reset_index(drop=True)


def threshold_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    flagged = probability >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    tn = int(np.sum(~flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    n = len(y)
    events = int(y.sum())
    odds = threshold / (1 - threshold)
    prevalence = events / n
    net_benefit = tp / n - fp / n * odds
    treat_all = prevalence - (1 - prevalence) * odds
    return {
        "threshold_probability": threshold,
        "n": n,
        "events": events,
        "prevalence": prevalence,
        "flagged_n": int(flagged.sum()),
        "flagged_proportion": float(flagged.mean()),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "sensitivity": tp / events if events else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "captured_events": tp,
        "captured_event_proportion": tp / events if events else np.nan,
        "net_benefit_model": net_benefit,
        "net_benefit_treat_all": treat_all,
        "net_benefit_treat_none": 0.0,
        "net_reduction_interventions_per_100_vs_treat_all": (
            (net_benefit - treat_all) / odds * 100
        ),
    }


def full_curve(frame: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame["outcome_na_ge_151"].to_numpy(dtype=int)
    probability = frame["predicted_risk"].to_numpy(dtype=float)
    metrics = pd.DataFrame([threshold_metrics(y, probability, float(pt)) for pt in GRID])
    metrics.insert(0, "dataset", label)
    curve = pd.concat(
        [
            metrics[["dataset", "threshold_probability", "n", "events", "prevalence", "net_benefit_model"]]
            .rename(columns={"net_benefit_model": "net_benefit"})
            .assign(strategy="model"),
            metrics[["dataset", "threshold_probability", "n", "events", "prevalence", "net_benefit_treat_all"]]
            .rename(columns={"net_benefit_treat_all": "net_benefit"})
            .assign(strategy="treat_all"),
            metrics[["dataset", "threshold_probability", "n", "events", "prevalence", "net_benefit_treat_none"]]
            .rename(columns={"net_benefit_treat_none": "net_benefit"})
            .assign(strategy="treat_none"),
        ],
        ignore_index=True,
    )
    representative = metrics.loc[
        metrics["threshold_probability"].isin(REPRESENTATIVE_THRESHOLDS)
    ].copy()
    return curve, representative


def bootstrap_ci(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    y = frame["outcome_na_ge_151"].to_numpy(dtype=int)
    probability = frame["predicted_risk"].to_numpy(dtype=float)
    point = pd.DataFrame(
        [threshold_metrics(y, probability, float(pt)) for pt in REPRESENTATIVE_THRESHOLDS]
    ).set_index("threshold_probability")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: dict[float, list[tuple[float, float]]] = {float(pt): [] for pt in REPRESENTATIVE_THRESHOLDS}
    for _ in range(BOOTSTRAP_RUNS):
        index = rng.integers(0, len(frame), len(frame))
        for threshold in REPRESENTATIVE_THRESHOLDS:
            result = threshold_metrics(y[index], probability[index], float(threshold))
            draws[float(threshold)].append(
                (
                    result["net_benefit_model"],
                    result["net_reduction_interventions_per_100_vs_treat_all"],
                )
            )
    rows = []
    for threshold, values in draws.items():
        values_array = np.asarray(values)
        rows.append(
            {
                "dataset": label,
                "threshold_probability": threshold,
                "net_benefit_model": point.loc[threshold, "net_benefit_model"],
                "net_benefit_model_ci95_low_bootstrap": float(np.quantile(values_array[:, 0], 0.025)),
                "net_benefit_model_ci95_high_bootstrap": float(np.quantile(values_array[:, 0], 0.975)),
                "net_reduction_interventions_per_100_vs_treat_all": point.loc[
                    threshold, "net_reduction_interventions_per_100_vs_treat_all"
                ],
                "net_reduction_ci95_low_bootstrap": float(np.quantile(values_array[:, 1], 0.025)),
                "net_reduction_ci95_high_bootstrap": float(np.quantile(values_array[:, 1], 0.975)),
                "bootstrap_runs": BOOTSTRAP_RUNS,
                "bootstrap_unit": "stay_id (one unique subject_id per temporal validation stay)",
            }
        )
    return pd.DataFrame(rows)


def plot_dca(
    curve: pd.DataFrame,
    bootstrap: pd.DataFrame | None,
    title: str,
    output_path: Path,
) -> None:
    colors = {"model": "#0072B2", "treat_all": "#D55E00", "treat_none": "#4D4D4D"}
    styles = {"model": "-", "treat_all": "--", "treat_none": ":"}
    labels = {"model": "27-feature LASSO (raw probability)", "treat_all": "Treat all", "treat_none": "Treat none"}
    figure, axis = plt.subplots(figsize=(9.2, 5.6), layout="constrained")
    for strategy in ["model", "treat_all", "treat_none"]:
        subset = curve.loc[curve["strategy"] == strategy]
        axis.plot(
            subset["threshold_probability"] * 100,
            subset["net_benefit"],
            color=colors[strategy],
            linestyle=styles[strategy],
            linewidth=2.2 if strategy == "model" else 1.6,
            label=labels[strategy],
        )
    if bootstrap is not None:
        focus = bootstrap.loc[bootstrap["threshold_probability"].isin([0.03, 0.05, 0.075, 0.10])]
        axis.errorbar(
            focus["threshold_probability"] * 100,
            focus["net_benefit_model"],
            yerr=[
                focus["net_benefit_model"] - focus["net_benefit_model_ci95_low_bootstrap"],
                focus["net_benefit_model_ci95_high_bootstrap"] - focus["net_benefit_model"],
            ],
            fmt="o",
            color="#0072B2",
            capsize=3,
            markersize=4,
            label="Model 95% bootstrap CI",
        )
    axis.axhline(0, color="0.75", linewidth=0.8)
    axis.set_xlim(1, 15)
    axis.set_xlabel("Threshold probability (%)")
    axis.set_ylabel("Net benefit")
    axis.set_title(title, fontsize=11)
    axis.legend(frameon=False, fontsize=8, loc="upper right")
    axis.grid(axis="y", color="0.9", linewidth=0.8)
    figure.savefig(output_path, dpi=240)
    plt.close(figure)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    development_path = MODEL_DIR / "lasso_development_nested_oof_predictions.csv"
    temporal_path = MODEL_DIR / "lasso_temporal_test_predictions.csv"
    development = load_predictions(development_path, "development_oof")
    temporal = load_predictions(temporal_path, "temporal_validation")

    locked_data = pd.read_csv(DATA_PATH, usecols=["stay_id", "subject_id", "temporal_split"])
    temporal_subjects = locked_data.loc[
        locked_data["temporal_split"] == "temporal_test_2017_2022"
    ]
    if len(temporal_subjects) != len(temporal):
        raise ValueError("Temporal prediction n differs from locked temporal cohort")
    if temporal_subjects["stay_id"].nunique() != len(temporal_subjects):
        raise ValueError("Locked temporal cohort has non-unique stay_id")
    if temporal_subjects["subject_id"].nunique() != len(temporal_subjects):
        raise ValueError("Temporal validation has repeated subject_id; patient bootstrap requires clustering")
    if set(temporal_subjects["stay_id"]) != set(temporal["stay_id"]):
        raise ValueError("Temporal prediction stays differ from locked temporal cohort")
    if (len(temporal), int(temporal["outcome_na_ge_151"].sum())) != (11246, 363):
        raise ValueError("Temporal data no longer matches locked n/events")

    dev_curve, dev_representative = full_curve(development, "development_oof")
    temporal_curve, temporal_representative = full_curve(temporal, "temporal_validation")
    threshold_metrics_frame = pd.concat([dev_representative, temporal_representative], ignore_index=True)
    net_benefit_frame = pd.concat([dev_curve, temporal_curve], ignore_index=True)
    bootstrap_frame = bootstrap_ci(temporal, "temporal_validation")

    threshold_metrics_frame.to_csv(OUTPUT_DIR / "dca_threshold_metrics.csv", index=False)
    net_benefit_frame.to_csv(OUTPUT_DIR / "dca_net_benefit.csv", index=False)
    bootstrap_frame.to_csv(OUTPUT_DIR / "dca_bootstrap_ci.csv", index=False)
    plot_dca(
        temporal_curve,
        bootstrap_frame,
        "Temporal validation DCA: locked raw probabilities",
        OUTPUT_DIR / "temporal_validation_dca.png",
    )
    plot_dca(
        dev_curve,
        None,
        "Development nested OOF DCA: locked raw probabilities",
        OUTPUT_DIR / "development_oof_dca.png",
    )

    validation = {
        "status": "PASS",
        "analysis_type": "decision_curve_analysis_only",
        "model": "locked_27_feature_primary_lasso",
        "probability_scale": "frozen_raw_probability_no_recalibration",
        "threshold_grid": {"minimum": 0.01, "maximum": 0.15, "step": 0.001},
        "representative_thresholds": REPRESENTATIVE_THRESHOLDS.tolist(),
        "input_prediction_hashes": {
            "development_oof": file_hash(development_path),
            "temporal_validation": file_hash(temporal_path),
            "locked_primary_dataset": file_hash(DATA_PATH),
        },
        "data_integrity": {
            "development_oof": {
                "n": int(len(development)),
                "events": int(development["outcome_na_ge_151"].sum()),
                "unique_stay_id": int(development["stay_id"].nunique()),
            },
            "temporal_validation": {
                "n": int(len(temporal)),
                "events": int(temporal["outcome_na_ge_151"].sum()),
                "unique_stay_id": int(temporal["stay_id"].nunique()),
                "unique_subject_id": int(temporal_subjects["subject_id"].nunique()),
            },
        },
        "bootstrap": {
            "runs": BOOTSTRAP_RUNS,
            "seed": BOOTSTRAP_SEED,
            "unit": "stay_id; verified one unique subject_id per time-validation stay",
        },
        "prohibited_actions": {
            "retrain": False,
            "recalibration": False,
            "feature_selection": False,
            "cohort_change": False,
            "xgboost": False,
            "clinical_threshold_selection": False,
        },
    }
    (OUTPUT_DIR / "dca_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(threshold_metrics_frame.to_string(index=False))
    print(bootstrap_frame.to_string(index=False))
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
