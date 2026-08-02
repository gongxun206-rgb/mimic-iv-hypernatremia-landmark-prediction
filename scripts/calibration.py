"""Calibration-only audit for the locked 27-feature primary LASSO model."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from statsmodels.nonparametric.smoothers_lowess import lowess


PROJECT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT / "modeling" / "v1.0_locked_primary"
EXISTING_AUDIT_DIR = MODEL_DIR / "audit"
OUTPUT_DIR = MODEL_DIR / "calibration_closure_v1.0"
METRIC_BOOTSTRAPS = 1000
SMOOTH_BOOTSTRAPS = 200
METRIC_RNG_SEED = 20260730
SMOOTH_RNG_SEED = 20260731


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    x = logit(probability).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000).fit(x, y)
    return float(fitted.intercept_[0]), float(fitted.coef_[0][0])


def metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration(y, probability)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def bootstrap_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    draws: list[dict[str, float]] = []
    while len(draws) < METRIC_BOOTSTRAPS:
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size == 2:
            draws.append(metric_values(y[index], probability[index]))
    frame = pd.DataFrame(draws)
    return {
        metric: tuple(frame[metric].quantile([0.025, 0.975]))
        for metric in frame.columns
    }


def wilson_interval(events: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    proportion = events / n
    denominator = 1 + z**2 / n
    centre = (proportion + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(
        proportion * (1 - proportion) / n + z**2 / (4 * n**2)
    ) / denominator
    return float(max(0, centre - margin)), float(min(1, centre + margin))


def ranked_deciles(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    ordered = frame.sort_values(["predicted_risk", "stay_id"], kind="mergesort").copy()
    ordered["decile"] = pd.qcut(
        np.arange(len(ordered)), q=10, labels=np.arange(1, 11)
    ).astype(int)
    total_events = int(ordered["outcome_na_ge_151"].sum())
    rows = []
    for decile, group in ordered.groupby("decile", sort=True):
        n = int(len(group))
        events = int(group["outcome_na_ge_151"].sum())
        predicted = float(group["predicted_risk"].mean())
        observed = events / n
        low, high = wilson_interval(events, n)
        rows.append(
            {
                "dataset": label,
                "decile": int(decile),
                "risk_direction": "highest" if decile == 10 else "lowest" if decile == 1 else "middle",
                "n": n,
                "events": events,
                "event_capture_proportion": events / total_events if total_events else np.nan,
                "mean_predicted_risk": predicted,
                "observed_event_rate": observed,
                "observed_ci95_low_wilson": low,
                "observed_ci95_high_wilson": high,
                "observed_minus_predicted": observed - predicted,
                "observed_expected_ratio": observed / predicted if predicted > 0 else np.nan,
                "risk_min": float(group["predicted_risk"].min()),
                "risk_max": float(group["predicted_risk"].max()),
            }
        )
    return pd.DataFrame(rows)


def high_risk_summary(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["predicted_risk", "stay_id"], ascending=[False, True], kind="mergesort"
    )
    total_events = int(ordered["outcome_na_ge_151"].sum())
    rows = []
    for label_name, proportion in [("top_10_percent", 0.10), ("top_20_percent", 0.20)]:
        n = int(np.ceil(len(ordered) * proportion))
        group = ordered.iloc[:n]
        events = int(group["outcome_na_ge_151"].sum())
        observed = events / n
        predicted = float(group["predicted_risk"].mean())
        low, high = wilson_interval(events, n)
        rows.append(
            {
                "dataset": label,
                "risk_group": label_name,
                "target_population_proportion": proportion,
                "n": n,
                "events": events,
                "captured_event_proportion": events / total_events if total_events else np.nan,
                "mean_predicted_risk": predicted,
                "observed_event_rate": observed,
                "observed_ci95_low_wilson": low,
                "observed_ci95_high_wilson": high,
                "observed_minus_predicted": observed - predicted,
                "observed_expected_ratio": observed / predicted if predicted > 0 else np.nan,
                "risk_threshold_inclusive": float(group["predicted_risk"].min()),
            }
        )
    return pd.DataFrame(rows)


def smooth_fit(probability: np.ndarray, outcome: np.ndarray, grid: np.ndarray) -> np.ndarray:
    fitted = lowess(
        endog=outcome,
        exog=probability,
        frac=0.30,
        it=0,
        delta=0.005 * max(float(probability.max() - probability.min()), 1e-6),
        return_sorted=True,
    )
    x_unique, inverse = np.unique(fitted[:, 0], return_inverse=True)
    y_unique = np.bincount(inverse, weights=fitted[:, 1]) / np.bincount(inverse)
    return np.interp(grid, x_unique, y_unique)


def smooth_curve(frame: pd.DataFrame, label: str, seed_offset: int) -> pd.DataFrame:
    probability = frame["predicted_risk"].to_numpy(dtype=float)
    outcome = frame["outcome_na_ge_151"].to_numpy(dtype=float)
    grid = np.quantile(probability, np.linspace(0.01, 0.99, 100))
    estimate = smooth_fit(probability, outcome, grid)
    rng = np.random.default_rng(SMOOTH_RNG_SEED + seed_offset)
    draws = []
    for _ in range(SMOOTH_BOOTSTRAPS):
        index = rng.integers(0, len(frame), len(frame))
        draws.append(smooth_fit(probability[index], outcome[index], grid))
    bands = np.quantile(np.asarray(draws), [0.025, 0.975], axis=0)
    return pd.DataFrame(
        {
            "dataset": label,
            "mean_predicted_risk": grid,
            "smoothed_observed_risk": estimate,
            "smoothed_ci95_low_bootstrap": bands[0],
            "smoothed_ci95_high_bootstrap": bands[1],
            "smoother": "LOWESS_frac_0.30",
            "smooth_bootstrap_runs": SMOOTH_BOOTSTRAPS,
        }
    )


def plot_calibration(
    deciles: pd.DataFrame,
    smooth: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    max_axis = max(
        0.02,
        float(
            max(
                deciles["mean_predicted_risk"].max(),
                deciles["observed_event_rate"].max(),
                smooth["smoothed_observed_risk"].max(),
            )
            * 1.15
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for axis in axes:
        axis.plot([0, max_axis], [0, max_axis], "--", color="0.5", label="ideal")
        axis.set_xlim(0, max_axis)
        axis.set_ylim(0, max_axis)
        axis.set_xlabel("Mean predicted risk")
        axis.set_ylabel("Observed event rate")
        axis.tick_params(labelsize=8)

    axes[0].errorbar(
        deciles["mean_predicted_risk"],
        deciles["observed_event_rate"],
        yerr=[
            deciles["observed_event_rate"] - deciles["observed_ci95_low_wilson"],
            deciles["observed_ci95_high_wilson"] - deciles["observed_event_rate"],
        ],
        fmt="o-",
        color="#1f77b4",
        capsize=3,
        label="decile observed (Wilson 95% CI)",
    )
    axes[0].set_title(f"{title}: decile calibration", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].fill_between(
        smooth["mean_predicted_risk"],
        smooth["smoothed_ci95_low_bootstrap"],
        smooth["smoothed_ci95_high_bootstrap"],
        color="#ff7f0e",
        alpha=0.20,
        label="200-bootstrap band",
    )
    axes[1].plot(
        smooth["mean_predicted_risk"],
        smooth["smoothed_observed_risk"],
        color="#ff7f0e",
        label="LOWESS observed calibration",
    )
    axes[1].set_title(f"{title}: smooth calibration", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    prediction_paths = {
        "logistic_development_oof": MODEL_DIR / "logistic_development_oof_predictions.csv",
        "logistic_temporal_test": MODEL_DIR / "logistic_temporal_test_predictions.csv",
        "lasso_development_oof": MODEL_DIR / "lasso_development_nested_oof_predictions.csv",
        "lasso_temporal_validation": MODEL_DIR / "lasso_temporal_test_predictions.csv",
    }
    predictions = {name: pd.read_csv(path) for name, path in prediction_paths.items()}
    for name, frame in predictions.items():
        if frame["stay_id"].duplicated().any():
            raise ValueError(f"{name}: duplicate stay_id")
        if not np.isfinite(frame["predicted_risk"]).all():
            raise ValueError(f"{name}: non-finite predicted risk")
        if not frame["predicted_risk"].between(0, 1).all():
            raise ValueError(f"{name}: predicted risk outside [0, 1]")

    # Reproduce the original script's exact RNG ordering before extracting LASSO rows.
    rng = np.random.default_rng(METRIC_RNG_SEED)
    metric_rows = []
    ordered_inputs = [
        ("logistic", "development_oof", predictions["logistic_development_oof"]),
        ("logistic", "temporal_test", predictions["logistic_temporal_test"]),
        ("lasso", "development_nested_oof", predictions["lasso_development_oof"]),
        ("lasso", "temporal_test", predictions["lasso_temporal_validation"]),
    ]
    for model, dataset, frame in ordered_inputs:
        y = frame["outcome_na_ge_151"].to_numpy()
        probability = frame["predicted_risk"].to_numpy()
        estimate = metric_values(y, probability)
        intervals = bootstrap_metrics(y, probability, rng)
        if model == "lasso":
            for metric in [
                "calibration_intercept",
                "calibration_slope",
                "brier_score",
            ]:
                low, high = intervals[metric]
                metric_rows.append(
                    {
                        "dataset": (
                            "development_oof"
                            if dataset == "development_nested_oof"
                            else "temporal_validation"
                        ),
                        "metric": metric,
                        "estimate": estimate[metric],
                        "ci95_low_bootstrap": low,
                        "ci95_high_bootstrap": high,
                        "bootstrap_runs": METRIC_BOOTSTRAPS,
                    }
                )
    calibration_metrics = pd.DataFrame(metric_rows)
    calibration_metrics.to_csv(OUTPUT_DIR / "calibration_metrics.csv", index=False)

    existing = pd.read_csv(EXISTING_AUDIT_DIR / "performance_calibration_bootstrap_ci.csv")
    consistency_rows = []
    source_map = {
        "development_oof": "development_nested_oof",
        "temporal_validation": "temporal_test",
    }
    for _, row in calibration_metrics.iterrows():
        prior = existing.loc[
            (existing["model"] == "lasso")
            & (existing["dataset"] == source_map[row["dataset"]])
            & (existing["metric"] == row["metric"])
        ].iloc[0]
        differences = {
            "estimate": float(row["estimate"] - prior["estimate"]),
            "ci95_low": float(row["ci95_low_bootstrap"] - prior["ci95_low"]),
            "ci95_high": float(row["ci95_high_bootstrap"] - prior["ci95_high"]),
        }
        consistency_rows.append(
            {
                "dataset": row["dataset"],
                "metric": row["metric"],
                "existing_estimate": prior["estimate"],
                "recomputed_estimate": row["estimate"],
                "existing_ci95_low": prior["ci95_low"],
                "recomputed_ci95_low": row["ci95_low_bootstrap"],
                "existing_ci95_high": prior["ci95_high"],
                "recomputed_ci95_high": row["ci95_high_bootstrap"],
                "max_abs_difference": max(abs(value) for value in differences.values()),
                "consistent_exactly": max(abs(value) for value in differences.values()) < 1e-12,
            }
        )
    consistency = pd.DataFrame(consistency_rows)
    consistency.to_csv(OUTPUT_DIR / "calibration_recalculation_consistency.csv", index=False)
    if not consistency["consistent_exactly"].all():
        raise RuntimeError("Recomputed locked calibration metrics differ from existing audit")

    outputs = []
    for offset, (label, frame) in enumerate(
        [
            ("development_oof", predictions["lasso_development_oof"]),
            ("temporal_validation", predictions["lasso_temporal_validation"]),
        ]
    ):
        deciles = ranked_deciles(frame, label)
        high_risk = high_risk_summary(frame, label)
        smooth = smooth_curve(frame, label, offset)
        deciles.to_csv(OUTPUT_DIR / f"{label}_calibration_deciles.csv", index=False)
        high_risk.to_csv(OUTPUT_DIR / f"{label}_high_risk_group_summary.csv", index=False)
        smooth.to_csv(OUTPUT_DIR / f"{label}_smooth_calibration_curve.csv", index=False)
        figure_name = (
            "development_oof_calibration.png"
            if label == "development_oof"
            else "temporal_validation_calibration.png"
        )
        plot_calibration(
            deciles,
            smooth,
            "Development nested OOF" if label == "development_oof" else "Temporal validation",
            OUTPUT_DIR / figure_name,
        )
        outputs.extend([deciles, high_risk])

    combined_deciles = pd.concat(
        [
            pd.read_csv(OUTPUT_DIR / "development_oof_calibration_deciles.csv"),
            pd.read_csv(OUTPUT_DIR / "temporal_validation_calibration_deciles.csv"),
        ],
        ignore_index=True,
    )
    combined_high_risk = pd.concat(
        [
            pd.read_csv(OUTPUT_DIR / "development_oof_high_risk_group_summary.csv"),
            pd.read_csv(OUTPUT_DIR / "temporal_validation_high_risk_group_summary.csv"),
        ],
        ignore_index=True,
    )
    combined_deciles.to_csv(OUTPUT_DIR / "calibration_deciles.csv", index=False)
    combined_high_risk.to_csv(OUTPUT_DIR / "high_risk_group_summary.csv", index=False)

    validation = {
        "status": "PASS",
        "analysis_type": "calibration_assessment_only",
        "model": "locked_27_feature_primary_lasso",
        "model_refit": False,
        "cohort_or_feature_change": False,
        "input_prediction_hashes": {
            name: file_hash(path) for name, path in prediction_paths.items()
        },
        "metric_bootstrap_runs": METRIC_BOOTSTRAPS,
        "smooth_curve_bootstrap_runs": SMOOTH_BOOTSTRAPS,
        "metric_recalculation_exact_match": True,
        "prediction_integrity": {
            name: {
                "n": int(len(frame)),
                "events": int(frame["outcome_na_ge_151"].sum()),
                "unique_stay_id": int(frame["stay_id"].nunique()),
            }
            for name, frame in predictions.items()
        },
        "prohibited_actions": {
            "retrain": False,
            "recalibration": False,
            "dca": False,
            "xgboost": False,
            "threshold_selection": False,
        },
    }
    (OUTPUT_DIR / "calibration_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(calibration_metrics.to_string(index=False))
    print(combined_high_risk.to_string(index=False))
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
