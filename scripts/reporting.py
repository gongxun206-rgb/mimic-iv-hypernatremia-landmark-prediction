"""Generate submission-facing flow, Table 1, and missingness materials."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import yaml


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT / "data" / "primary_model_dataset.csv"
DEFAULT_FEATURES = PROJECT / "config" / "predictors.yaml"
DEFAULT_PREPROCESSING = PROJECT / "model_specification" / "preprocessing_parameters.csv"
DEFAULT_FLOW = PROJECT / "config" / "locked_flow_counts.csv"
DEFAULT_OUTPUT = PROJECT / "outputs" / "submission_materials"

EXPECTED = {
    "n": 34913,
    "events": 823,
    "development_n": 23667,
    "development_events": 460,
    "temporal_n": 11246,
    "temporal_events": 363,
}

LABELS = {
    "age": "Age, years",
    "sodium_last": "Last sodium before T0, mmol/L",
    "creatinine_last": "Last creatinine before T0, mg/dL",
    "bun_last": "Last BUN before T0, mg/dL",
    "chloride_last": "Last chloride before T0, mmol/L",
    "potassium_last": "Last potassium before T0, mmol/L",
    "bicarbonate_last": "Last bicarbonate before T0, mmol/L",
    "glucose_last": "Last glucose before T0, mg/dL",
    "heart_rate_last": "Last heart rate before T0, beats/min",
    "mbp_last": "Last mean blood pressure before T0, mmHg",
    "resp_rate_last": "Last respiratory rate before T0, breaths/min",
    "spo2_last": "Last SpO2 before T0, %",
    "temperature_last": "Last temperature before T0, °C",
    "uo_mlkghr_6hr_clean": "Cleaned 6-h urine output rate before T0, mL/kg/h",
    "uo_mlkghr_12hr_clean": "Cleaned 12-h urine output rate before T0, mL/kg/h",
    "gender": "Female sex",
    "invasive_vent_pre_t0": "Invasive ventilation before T0",
    "noninvasive_vent_pre_t0": "Non-invasive ventilation before T0",
    "hfnc_pre_t0": "High-flow nasal cannula before T0",
    "supplemental_oxygen_pre_t0": "Supplemental oxygen before T0",
    "tracheostomy_pre_t0": "Tracheostomy before T0",
    "ventilation_no_record_pre_t0": "No qualifying respiratory-support record before T0",
    "loop_diuretic_pre_t0": "Loop diuretic before T0",
    "vasoactive_pre_t0": "Vasoactive agent before T0",
    "rrt_pre_t0": "Renal replacement therapy before T0",
    "iv_isotonic_crystalloid_pre_t0": "IV isotonic crystalloid before T0",
    "dextrose_or_hypotonic_fluid_pre_t0": "Dextrose or hypotonic fluid before T0",
}

NUMERIC = list(LABELS)[:15]
BINARY = list(LABELS)[16:]


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "t", "1"])


def smd_continuous(development: pd.Series, temporal: pd.Series) -> float:
    development = development.dropna().astype(float)
    temporal = temporal.dropna().astype(float)
    denominator = np.sqrt((development.var(ddof=1) + temporal.var(ddof=1)) / 2)
    return float((temporal.mean() - development.mean()) / denominator) if denominator else 0.0


def smd_binary(development: pd.Series, temporal: pd.Series) -> float:
    p_development, p_temporal = float(development.mean()), float(temporal.mean())
    denominator = np.sqrt(
        (p_development * (1 - p_development) + p_temporal * (1 - p_temporal)) / 2
    )
    return float((p_temporal - p_development) / denominator) if denominator else 0.0


def median_iqr(series: pd.Series) -> str:
    values = series.dropna().astype(float)
    return f"{values.median():.1f} [{values.quantile(.25):.1f}, {values.quantile(.75):.1f}]"


def count_pct(series: pd.Series) -> str:
    return f"{int(series.sum())} ({100 * series.mean():.1f})"


def add_binary_row(rows: list[dict], label: str, variable: str, level: str,
                   overall: pd.Series, development: pd.Series, temporal: pd.Series,
                   section: str) -> None:
    rows.append({
        "section": section,
        "characteristic": label,
        "variable": variable,
        "level": level,
        "summary_type": "n (%)",
        "overall": count_pct(overall),
        "development": count_pct(development),
        "temporal_validation": count_pct(temporal),
        "smd_value": smd_binary(development, temporal),
    })


def build_table1(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    development = data.loc[data["temporal_split"] == "development_2008_2016"]
    temporal = data.loc[data["temporal_split"] == "temporal_test_2017_2022"]
    rows: list[dict] = []
    for feature in features:
        if feature in NUMERIC:
            rows.append({
                "section": "Core predictors",
                "characteristic": LABELS[feature],
                "variable": feature,
                "level": "",
                "summary_type": "median [IQR]",
                "overall": median_iqr(data[feature]),
                "development": median_iqr(development[feature]),
                "temporal_validation": median_iqr(temporal[feature]),
                "smd_value": smd_continuous(development[feature], temporal[feature]),
            })
        elif feature == "gender":
            add_binary_row(
                rows, LABELS[feature], feature, "F",
                data[feature].astype(str).eq("F"), development[feature].astype(str).eq("F"),
                temporal[feature].astype(str).eq("F"), "Core predictors",
            )
        else:
            add_binary_row(
                rows, LABELS[feature], feature, "1", bool_series(data[feature]),
                bool_series(development[feature]), bool_series(temporal[feature]), "Core predictors",
            )
    for level in sorted(data["first_careunit"].dropna().unique()):
        add_binary_row(
            rows, f"First care unit recorded for the ICU stay: {level}", "first_careunit", level,
            data["first_careunit"].eq(level), development["first_careunit"].eq(level),
            temporal["first_careunit"].eq(level), "First care unit",
        )
    table = pd.DataFrame(rows)
    table["smd"] = table["smd_value"].map(lambda value: f"{value:.3f}")
    return table


def build_missingness(data: pd.DataFrame, features: list[str], preprocessing: pd.DataFrame) -> pd.DataFrame:
    development = data.loc[data["temporal_split"] == "development_2008_2016"]
    temporal = data.loc[data["temporal_split"] == "temporal_test_2017_2022"]
    indicators = set(
        preprocessing.loc[preprocessing["missing_indicator_generated"].astype(str).str.lower() == "true", "original_variable"]
    )
    rows = []
    for feature in features:
        counts = [int(frame[feature].isna().sum()) for frame in (data, development, temporal)]
        percentages = [100 * count / len(frame) for count, frame in zip(counts, (data, development, temporal))]
        rows.append({
            "feature": feature,
            "feature_label": LABELS[feature],
            "overall_n": len(data), "overall_missing_n": counts[0], "overall_missing_pct": percentages[0],
            "development_n": len(development), "development_missing_n": counts[1], "development_missing_pct": percentages[1],
            "temporal_validation_n": len(temporal), "temporal_validation_missing_n": counts[2], "temporal_validation_missing_pct": percentages[2],
            "temporal_minus_development_missing_pct_points": percentages[2] - percentages[1],
            "missing_indicator_in_final_pipeline": feature in indicators,
        })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = ["Characteristic", "Overall (N=34,913)", "Development 2008-2016 (N=23,667)", "Temporal validation 2017-2022 (N=11,246)", "SMD"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['characteristic']} | {row['overall']} | {row['development']} | "
            f"{row['temporal_validation']} | {row['smd']} |"
        )
    return "\n".join(lines)


def plot_flow(flow: pd.DataFrame, path: Path) -> None:
    display = flow.loc[flow["node_order"].between(1, 6)].copy()
    figure, axis = plt.subplots(figsize=(8.5, 13), layout="constrained")
    axis.set(xlim=(0, 1), ylim=(0, len(display) + 0.5))
    axis.axis("off")
    previous_n = None
    for index, (_, row) in enumerate(display.iterrows()):
        y = len(display) - index - 0.2
        text = f"{row['node']}\nn = {int(row['n_stays']):,}"
        if pd.notna(row["no_followup_sodium"]) and int(row["no_followup_sodium"]) > 0:
            text += f"\nNo post-T0 sodium: {int(row['no_followup_sodium']):,}"
        if previous_n is not None and previous_n != int(row["n_stays"]):
            text += f"\nExcluded since prior node: {previous_n - int(row['n_stays']):,}"
        previous_n = int(row["n_stays"])
        box = FancyBboxPatch((.12, y - .64), .76, .78, boxstyle="round,pad=0.02",
                             facecolor="#F3F6F8", edgecolor="#294E6B", linewidth=1.2)
        axis.add_patch(box)
        axis.text(.5, y - .25, text, ha="center", va="center", fontsize=10)
        if index < len(display) - 1:
            axis.annotate("", xy=(.5, y - .86), xytext=(.5, y - .67),
                          arrowprops={"arrowstyle": "->", "color": "#294E6B", "lw": 1.2})
    axis.text(.5, .08, "Development 23,667 (460 events); temporal validation 11,246 (363 events)",
              ha="center", va="center", fontsize=9)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def validate_locked_counts(data: pd.DataFrame) -> dict:
    outcome = bool_series(data["outcome_na_ge_151"])
    development = data["temporal_split"].eq("development_2008_2016")
    temporal = data["temporal_split"].eq("temporal_test_2017_2022")
    observed = {
        "n": len(data), "events": int(outcome.sum()),
        "development_n": int(development.sum()), "development_events": int(outcome[development].sum()),
        "temporal_n": int(temporal.sum()), "temporal_events": int(outcome[temporal].sum()),
    }
    if observed != EXPECTED:
        raise ValueError(f"Locked primary counts differ: {observed}; expected {EXPECTED}")
    if data["stay_id"].duplicated().any():
        raise ValueError("stay_id is not unique")
    return observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--preprocessing", type=Path, default=DEFAULT_PREPROCESSING)
    parser.add_argument("--flow-counts", type=Path, default=DEFAULT_FLOW)
    parser.add_argument(
        "--care-unit-map", type=Path,
        help="Optional authorized local CSV with stay_id and first_careunit when absent from input",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [args.input, args.features, args.preprocessing, args.flow_counts]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)

    data = pd.read_csv(args.input)
    if "first_careunit" not in data.columns and args.care_unit_map:
        care_units = pd.read_csv(args.care_unit_map, usecols=["stay_id", "first_careunit"])
        care_units = care_units.drop_duplicates("stay_id")
        data = data.merge(care_units, on="stay_id", how="left", validate="one_to_one")
    features = yaml.safe_load(args.features.read_text(encoding="utf-8"))["core_predictors"]
    if len(features) != 27 or len(set(features)) != 27:
        raise ValueError("Predictor manifest must contain 27 unique variables")
    missing = sorted(set(features + ["stay_id", "first_careunit", "temporal_split", "outcome_na_ge_151"]) - set(data.columns))
    if missing:
        raise ValueError(f"Required reporting columns are missing: {missing}")
    counts = validate_locked_counts(data)

    flow = pd.read_csv(args.flow_counts)
    flow.to_csv(output / "cohort_flow_counts_v1.0.csv", index=False)
    plot_flow(flow, output / "Figure_1_cohort_flow_v1.0.png")
    table1 = build_table1(data, features)
    table1.to_csv(output / "table1_primary_cohort_v1.0.csv", index=False)
    (output / "Table_1_primary_cohort_v1.0.md").write_text(
        "# Table 1. Primary cohort characteristics\n\n"
        "Values are median [IQR] or n (%). SMD compares temporal validation with development. "
        "The outcome is intentionally not included as a Table 1 row.\n\n" + markdown_table(table1) + "\n",
        encoding="utf-8",
    )
    preprocessing = pd.read_csv(args.preprocessing)
    missingness = build_missingness(data, features, preprocessing)
    missingness.to_csv(output / "Table_S2_missingness_v1.0.csv", index=False)

    generated = sorted(path for path in output.iterdir() if path.is_file())
    validation = {
        "status": "PASS",
        "locked_counts": counts,
        "feature_count": len(features),
        "table1_outcome_row_present": False,
        "first_careunit_label": "First care unit recorded for the ICU stay",
        "input_hashes": {
            **{path.name: file_hash(path) for path in paths},
            **({args.care_unit_map.name: file_hash(args.care_unit_map)} if args.care_unit_map else {}),
        },
        "output_hashes": {path.name: file_hash(path) for path in generated},
        "model_retrained": False,
    }
    (output / "submission_materials_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
