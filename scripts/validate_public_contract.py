"""Statically validate the public SQL-to-model and frozen-model contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd
import yaml


PROJECT = Path(__file__).resolve().parents[1]


def ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs" / "public_contract_validation.json")
    args = parser.parse_args()
    sql = (PROJECT / "sql" / "02_modeling_dataset_derivation.sql").read_text(encoding="utf-8")
    predictors = yaml.safe_load((PROJECT / "config" / "predictors.yaml").read_text(encoding="utf-8"))["core_predictors"]
    order = pd.read_csv(PROJECT / "model_specification" / "final_feature_order.csv").sort_values("transform_order")
    metadata = json.loads((PROJECT / "model_specification" / "model_metadata.json").read_text(encoding="utf-8"))
    match = re.search(
        r"\\copy \(SELECT (?P<columns>.*?) FROM hypernatremia_modeling_dataset ORDER BY stay_id\) "
        r"TO 'data/primary_model_dataset\.csv'",
        sql,
    )
    if not match:
        raise ValueError("Unable to locate primary SQL export")
    exported = [column.strip() for column in match.group("columns").split(",")]
    frozen_original = ordered_unique(order["original_variable"].tolist())
    forbidden_future = {
        "sodium_n_followup", "first_na_ge_151_time", "observation_end", "t0_time",
        "chemistry_last_charttime", "vital_last_charttime",
    }
    checks = {
        "predictor_manifest_has_27_unique": len(predictors) == 27 and len(set(predictors)) == 27,
        "all_predictors_exported_by_sql": set(predictors).issubset(exported),
        "frozen_original_order_matches_manifest": frozen_original == predictors,
        "no_forbidden_future_predictor_export": not forbidden_future.intersection(exported),
        "outcome_exported": "outcome_na_ge_151" in exported,
        "temporal_split_exported": "temporal_split" in exported,
        "gender_exported": "gender" in exported,
        "cleaned_urine_fields_exported": {"uo_mlkghr_6hr_clean", "uo_mlkghr_12hr_clean"}.issubset(exported),
        "high_certainty_export_present": "data/high_certainty_model_dataset.csv" in sql,
        "gu_itemids_both_checked": "o.itemid IN (227488, 227489)" in sql,
        "frozen_transformed_feature_count_41": len(order) == 41 == int(metadata["transformed_feature_count"]),
        "frozen_final_C_0_3": float(metadata["lasso_final_C"]) == 0.3,
        "frozen_intercept_matches": float(metadata["intercept"]) == -3.707283461539647,
    }
    result = {
        "status": "SQL_TO_MODEL_SCHEMA_PASS" if all(checks.values()) else "REVISION_REQUIRED",
        "checks": checks,
        "exported_columns": exported,
        "core_predictors": predictors,
        "model_input_uses_post_t0_predictor": False,
        "database_query_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "SQL_TO_MODEL_SCHEMA_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
