"""Create submission-facing flow, Table 1, and missingness outputs from locked v1.0 inputs."""

from __future__ import annotations

from hashlib import sha256
from io import StringIO
from pathlib import Path
import json
import os
import subprocess

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data" / "primary_cohort_model_dataset_v1.0.csv"
MANIFEST_PATH = PROJECT / "data" / "cohort_lock_manifest_v1.0.json"
FEATURE_MANIFEST_PATH = PROJECT / "modeling" / "v1.0_locked_primary" / "core_feature_manifest.csv"
MODEL_PATH = PROJECT / "modeling" / "v1.0_locked_primary" / "lasso_final_development.joblib"
FLOW_SQL_PATH = PROJECT / "sql" / "20_submission_cohort_flow_v1.0.sql"
OUTPUT_DIR = PROJECT / "paper_submission_materials" / "v1.0"
# Connection values are supplied by the caller's environment. Do not add a
# local host name, a user name, or a credential to this public repository.
PSQL_PATH = os.environ.get("PSQL_BIN", "psql")
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]


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
    "temperature_last": "Last temperature before T0, C",
    "uo_mlkghr_6hr_clean": "Clean 6-h urine output rate before T0, mL/kg/h",
    "uo_mlkghr_12hr_clean": "Clean 12-h urine output rate before T0, mL/kg/h",
    "gender": "Female sex",
    "invasive_vent_pre_t0": "Invasive ventilation before T0",
    "noninvasive_vent_pre_t0": "Non-invasive ventilation before T0",
    "hfnc_pre_t0": "High-flow nasal cannula before T0",
    "supplemental_oxygen_pre_t0": "Supplemental oxygen before T0",
    "tracheostomy_pre_t0": "Tracheostomy before T0",
    "ventilation_no_record_pre_t0": "No ventilation record before T0",
    "loop_diuretic_pre_t0": "Loop diuretic before T0",
    "vasoactive_pre_t0": "Vasoactive agent before T0",
    "rrt_pre_t0": "Renal replacement therapy before T0",
    "iv_isotonic_crystalloid_pre_t0": "IV isotonic crystalloid before T0",
    "dextrose_or_hypotonic_fluid_pre_t0": "Dextrose or hypotonic fluid before T0",
}


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "t", "1"])


def run_psql_sql_file(path: Path) -> pd.DataFrame:
    command = [
        str(PSQL_PATH), "-h", DB_HOST, "-p", DB_PORT, "-U", DB_USER, "-d", DB_NAME,
        "-X", "-q", "--csv", "-v", "ON_ERROR_STOP=1", "-f", str(path),
    ]
    completed = subprocess.run(
        command, check=True, text=True, encoding="utf-8", errors="replace",
        capture_output=True, env=os.environ.copy()
    )
    return pd.read_csv(StringIO(completed.stdout))


def run_psql_query(query: str) -> pd.DataFrame:
    command = [
        str(PSQL_PATH), "-h", DB_HOST, "-p", DB_PORT, "-U", DB_USER, "-d", DB_NAME,
        "-X", "-q", "--csv", "-v", "ON_ERROR_STOP=1", "-c", query,
    ]
    completed = subprocess.run(
        command, check=True, text=True, encoding="utf-8", errors="replace",
        capture_output=True, env=os.environ.copy()
    )
    return pd.read_csv(StringIO(completed.stdout))


def smd_continuous(development: pd.Series, temporal: pd.Series) -> float:
    development = development.dropna().astype(float)
    temporal = temporal.dropna().astype(float)
    denominator = np.sqrt((development.var(ddof=1) + temporal.var(ddof=1)) / 2)
    return float((temporal.mean() - development.mean()) / denominator) if denominator else 0.0


def smd_binary(development: pd.Series, temporal: pd.Series) -> float:
    p_development = float(development.mean())
    p_temporal = float(temporal.mean())
    denominator = np.sqrt((p_development * (1 - p_development) + p_temporal * (1 - p_temporal)) / 2)
    return float((p_temporal - p_development) / denominator) if denominator else 0.0


def format_median_iqr(series: pd.Series) -> str:
    series = series.dropna().astype(float)
    return f"{series.median():.1f} [{series.quantile(0.25):.1f}, {series.quantile(0.75):.1f}]"


def format_count_pct(series: pd.Series) -> str:
    n = int(series.sum())
    return f"{n} ({100 * n / len(series):.1f})"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = ["Characteristic", "Overall", "Development 2008-2016", "Temporal validation 2017-2022", "SMD"]
    output = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        output.append(
            "| " + " | ".join(
                [str(row[column]) for column in ["characteristic", "overall", "development", "temporal_validation", "smd"]]
            ) + " |"
        )
    return "\n".join(output)


def build_table1(data: pd.DataFrame, feature_manifest: pd.DataFrame, icu: pd.DataFrame) -> pd.DataFrame:
    data = data.merge(icu, on="stay_id", how="left", validate="one_to_one")
    if data["icu_type"].isna().any():
        raise ValueError("Missing ICU type after locked stay_id merge")
    development = data.loc[data["temporal_split"] == "development_2008_2016"].copy()
    temporal = data.loc[data["temporal_split"] == "temporal_test_2017_2022"].copy()
    rows = []
    for _, spec in feature_manifest.iterrows():
        feature = spec["feature"]
        label = LABELS[feature]
        if spec["feature_type"] == "numeric":
            rows.append({
                "section": "Core predictors", "characteristic": label, "variable": feature,
                "level": "", "summary_type": "median [IQR]", "overall": format_median_iqr(data[feature]),
                "development": format_median_iqr(development[feature]), "temporal_validation": format_median_iqr(temporal[feature]),
                "smd_value": smd_continuous(development[feature], temporal[feature]),
            })
        elif feature == "gender":
            overall = data[feature].astype(str).eq("F")
            dev = development[feature].astype(str).eq("F")
            test = temporal[feature].astype(str).eq("F")
            rows.append({
                "section": "Core predictors", "characteristic": label, "variable": feature, "level": "F",
                "summary_type": "n (%)", "overall": format_count_pct(overall), "development": format_count_pct(dev),
                "temporal_validation": format_count_pct(test), "smd_value": smd_binary(dev, test),
            })
        else:
            overall = bool_series(data[feature])
            dev = bool_series(development[feature])
            test = bool_series(temporal[feature])
            rows.append({
                "section": "Core predictors", "characteristic": label, "variable": feature, "level": "Yes",
                "summary_type": "n (%)", "overall": format_count_pct(overall), "development": format_count_pct(dev),
                "temporal_validation": format_count_pct(test), "smd_value": smd_binary(dev, test),
            })
    for level in sorted(data["icu_type"].unique()):
        overall = data["icu_type"].eq(level)
        dev = development["icu_type"].eq(level)
        test = temporal["icu_type"].eq(level)
        rows.append({
            "section": "ICU type", "characteristic": f"ICU type: {level}", "variable": "icu_type", "level": level,
            "summary_type": "n (%)", "overall": format_count_pct(overall), "development": format_count_pct(dev),
            "temporal_validation": format_count_pct(test), "smd_value": smd_binary(dev, test),
        })
    overall = bool_series(data["outcome_na_ge_151"])
    dev = bool_series(development["outcome_na_ge_151"])
    test = bool_series(temporal["outcome_na_ge_151"])
    rows.append({
        "section": "Outcome", "characteristic": "Recorded Na >=151 mmol/L", "variable": "outcome_na_ge_151", "level": "Yes",
        "summary_type": "n (%)", "overall": format_count_pct(overall), "development": format_count_pct(dev),
        "temporal_validation": format_count_pct(test), "smd_value": smd_binary(dev, test),
    })
    table = pd.DataFrame(rows)
    table["smd"] = table["smd_value"].map(lambda value: f"{value:.3f}")
    return table


def build_missingness(data: pd.DataFrame, features: list[str], pipeline) -> pd.DataFrame:
    development = data.loc[data["temporal_split"] == "development_2008_2016"]
    temporal = data.loc[data["temporal_split"] == "temporal_test_2017_2022"]
    numeric_features = pipeline.named_steps["preprocess"].transformers_[0][2]
    indicator_positions = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].indicator_.features_
    indicator_features = {numeric_features[position] for position in indicator_positions}
    rows = []
    for feature in features:
        overall_missing = int(data[feature].isna().sum())
        development_missing = int(development[feature].isna().sum())
        temporal_missing = int(temporal[feature].isna().sum())
        rows.append({
            "feature": feature,
            "feature_label": LABELS[feature],
            "overall_n": len(data),
            "overall_missing_n": overall_missing,
            "overall_missing_pct": 100 * overall_missing / len(data),
            "development_n": len(development),
            "development_missing_n": development_missing,
            "development_missing_pct": 100 * development_missing / len(development),
            "temporal_validation_n": len(temporal),
            "temporal_validation_missing_n": temporal_missing,
            "temporal_validation_missing_pct": 100 * temporal_missing / len(temporal),
            "temporal_minus_development_missing_pct_points": 100 * temporal_missing / len(temporal) - 100 * development_missing / len(development),
            "missing_indicator_in_final_pipeline": feature in indicator_features,
        })
    return pd.DataFrame(rows)


def plot_flow(flow: pd.DataFrame, path: Path) -> None:
    display = flow.loc[flow["node_order"].between(1, 7)].copy()
    labels = []
    for _, row in display.iterrows():
        text = f"{row['node']}\nn = {int(row['n_stays']):,}"
        if pd.notna(row["excluded_since_prior_node"]):
            text += f"\nExcluded since prior node: {int(row['excluded_since_prior_node']):,}"
        if pd.notna(row["recorded_events"]) and row["node_order"] >= 5:
            text += f"\nRecorded Na >=151: {int(row['recorded_events']):,}"
        if pd.notna(row["no_followup_sodium"]) and int(row["no_followup_sodium"]) > 0:
            text += f"\nNo follow-up Na: {int(row['no_followup_sodium']):,}"
        labels.append(text)
    figure, axis = plt.subplots(figsize=(8.5, 13), layout="constrained")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, len(labels) + 0.5)
    axis.axis("off")
    for index, label in enumerate(labels):
        y = len(labels) - index - 0.2
        box = FancyBboxPatch((0.12, y - 0.64), 0.76, 0.78, boxstyle="round,pad=0.02", facecolor="#F3F6F8", edgecolor="#294E6B", linewidth=1.2)
        axis.add_patch(box)
        axis.text(0.5, y - 0.25, label, ha="center", va="center", fontsize=10)
        if index < len(labels) - 1:
            axis.annotate("", xy=(0.5, y - 0.86), xytext=(0.5, y - 0.67), arrowprops={"arrowstyle": "->", "color": "#294E6B", "lw": 1.2})
    axis.text(0.5, 0.08, "Final binary cohort: development 23,667 (460 events); temporal validation 11,246 (363 events)", ha="center", va="center", fontsize=9)
    figure.savefig(path, dpi=260)
    plt.close(figure)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")
    if not PSQL_PATH.exists():
        raise FileNotFoundError(PSQL_PATH)
    if not os.environ.get("PGPASSWORD"):
        raise RuntimeError("Set PGPASSWORD before running the read-only PostgreSQL queries")
    OUTPUT_DIR.mkdir(parents=True)

    data = pd.read_csv(DATA_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    primary = next(item for item in manifest if item["role"] == "locked_primary")
    feature_manifest = pd.read_csv(FEATURE_MANIFEST_PATH)
    features = feature_manifest["feature"].tolist()
    if len(features) != 27 or len(set(features)) != 27:
        raise ValueError("Locked feature manifest is not 27 unique features")
    if data["stay_id"].duplicated().any() or len(data) != 34913:
        raise ValueError("Locked primary data stay_id/n integrity failed")
    outcome = bool_series(data["outcome_na_ge_151"])
    development = data["temporal_split"].eq("development_2008_2016")
    temporal = data["temporal_split"].eq("temporal_test_2017_2022")
    if (int(outcome.sum()), int(development.sum()), int(outcome[development].sum()), int(temporal.sum()), int(outcome[temporal].sum())) != (823, 23667, 460, 11246, 363):
        raise ValueError("Locked primary n/events/split integrity failed")
    if file_hash(DATA_PATH) != primary["sha256"]:
        raise ValueError("Locked primary hash differs from cohort manifest")

    flow = run_psql_sql_file(FLOW_SQL_PATH)
    flow["excluded_since_prior_node"] = np.nan
    sequential_index = flow.index[flow["node_order"].between(2, 7)]
    flow.loc[sequential_index, "excluded_since_prior_node"] = (
        flow.loc[sequential_index - 1, "n_stays"].to_numpy()
        - flow.loc[sequential_index, "n_stays"].to_numpy()
    )
    flow["flow_relation"] = np.where(
        flow["node_order"].between(1, 7),
        "sequential primary-cohort flow",
        "mutually exclusive temporal split of final binary modeling cohort",
    )
    expected = {6: (36324, 823, 1411), 7: (34913, 823, 0), 8: (23667, 460, 0), 9: (11246, 363, 0)}
    for node_order, values in expected.items():
        row = flow.loc[flow["node_order"] == node_order].iloc[0]
        observed = (int(row["n_stays"]), int(row["recorded_events"]), int(row["no_followup_sodium"]))
        if observed != values:
            raise ValueError(f"Flow mismatch at node {node_order}: {observed}, expected {values}")
    flow.to_csv(OUTPUT_DIR / "cohort_flow_counts_v1.0.csv", index=False)
    plot_flow(flow, OUTPUT_DIR / "Figure_1_cohort_flow_v1.0.png")

    icu = run_psql_query("SELECT stay_id, first_careunit AS icu_type FROM mimiciv_icu.icustays ORDER BY stay_id")
    if icu["stay_id"].duplicated().any():
        raise ValueError("ICU stay_id is not unique")
    table1 = build_table1(data, feature_manifest, icu)
    table1.to_csv(OUTPUT_DIR / "table1_primary_cohort_v1.0.csv", index=False)
    (OUTPUT_DIR / "Table_1_primary_cohort_v1.0.md").write_text(
        "# Table 1. Locked primary cohort characteristics\n\n"
        "Values are median [IQR] for continuous variables and n (%) for categorical variables. "
        "SMD is temporal validation minus development; it is calculated from means/pooled standard deviations for continuous variables and proportions for binary category indicators. P values are intentionally omitted.\n\n"
        + markdown_table(table1) + "\n",
        encoding="utf-8",
    )

    pipeline = joblib.load(MODEL_PATH)
    missingness = build_missingness(data, features, pipeline)
    missingness.to_csv(OUTPUT_DIR / "Table_S2_missingness_v1.0.csv", index=False)

    outputs = [
        OUTPUT_DIR / "cohort_flow_counts_v1.0.csv",
        OUTPUT_DIR / "Figure_1_cohort_flow_v1.0.png",
        OUTPUT_DIR / "table1_primary_cohort_v1.0.csv",
        OUTPUT_DIR / "Table_1_primary_cohort_v1.0.md",
        OUTPUT_DIR / "Table_S2_missingness_v1.0.csv",
    ]
    validation = {
        "status": "PASS",
        "analysis_type": "locked_data_submission_materials_only",
        "input_hashes": {
            "locked_primary_dataset": file_hash(DATA_PATH),
            "cohort_lock_manifest": file_hash(MANIFEST_PATH),
            "core_feature_manifest": file_hash(FEATURE_MANIFEST_PATH),
            "locked_lasso_pipeline": file_hash(MODEL_PATH),
            "flow_sql": file_hash(FLOW_SQL_PATH),
        },
        "qc": {
            "primary_n": len(data), "primary_events": int(outcome.sum()),
            "development_n": int(development.sum()), "development_events": int(outcome[development].sum()),
            "temporal_validation_n": int(temporal.sum()), "temporal_validation_events": int(outcome[temporal].sum()),
            "unique_stay_id": int(data["stay_id"].nunique()), "locked_feature_count": len(features),
            "flow_matches_locked_primary": True, "uses_v0_exploratory_input": False,
        },
        "prohibited_actions": {
            "model_retraining": False, "cohort_change": False, "feature_change": False,
            "outcome_change": False, "recalibration": False, "xgboost": False,
            "threshold_selection": False, "lasso_formula_export": False,
        },
        "output_hashes": {path.name: file_hash(path) for path in outputs},
    }
    (OUTPUT_DIR / "submission_materials_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(flow.to_string(index=False))
    print(table1[["characteristic", "overall", "development", "temporal_validation", "smd"]].to_string(index=False))
    print(missingness.to_string(index=False))
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
