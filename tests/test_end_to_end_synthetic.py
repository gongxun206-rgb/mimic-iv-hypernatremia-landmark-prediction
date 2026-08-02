from pathlib import Path

import numpy as np
import pandas as pd

from scripts.model_development import FEATURES, NUMERIC_FEATURES, make_pipeline, normalize_binary, outcome
from scripts.reporting import build_table1
from scripts.study_contract import cleaned_urine_rate, cohort_eligibility, recorded_outcome


ROOT = Path(__file__).resolve().parents[1]


def artificial_training_data() -> pd.DataFrame:
    base = pd.read_csv(ROOT / "synthetic_example" / "synthetic_example.csv")
    data = pd.concat([base] * 8, ignore_index=True)
    for feature in NUMERIC_FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce").astype(float)
        data[feature] = data[feature] + (np.arange(len(data)) % 3) * 0.001
    for row_index, feature in enumerate(NUMERIC_FEATURES[2:]):
        data.loc[row_index, feature] = np.nan
    return data


def test_preprocessing_and_small_lasso_fit():
    data = normalize_binary(artificial_training_data())
    assert len(FEATURES) == 27 and set(FEATURES).issubset(data.columns)
    forbidden = {"sodium_n_followup", "first_na_ge_151_time", "observation_end"}
    assert not forbidden.intersection(FEATURES)
    target = outcome(data["outcome_na_ge_151"])
    pipeline = make_pipeline("lasso", c_value=0.3)
    pipeline.fit(data[FEATURES], target)
    transformed = pipeline.named_steps["preprocess"].transform(data[FEATURES])
    probability = pipeline.predict_proba(data[FEATURES])[:, 1]
    assert transformed.shape[1] == 41
    assert probability.shape == (24,)
    assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))


def test_cohort_outcome_and_urine_contracts():
    assert cohort_eligibility([139]) == {"primary": True, "high_certainty": False}
    assert cohort_eligibility([138, 141]) == {"primary": True, "high_certainty": True}
    assert cohort_eligibility([134, 140]) == {"primary": False, "high_certainty": False}
    assert recorded_outcome([]) is None
    assert recorded_outcome([150, 151]) == 1
    candidates = [
        {"hours_before_t0": 1, "rate": 9.0, "weight": 70, "gu_itemids": [227489]},
        {"hours_before_t0": 2, "rate": 0.8, "weight": 70, "gu_itemids": []},
    ]
    assert cleaned_urine_rate(candidates, 6) == (0.8, None)
    assert cleaned_urine_rate(candidates[:1], 6) == (None, "GU_contaminated")


def test_public_sql_schema_contract():
    sql = (ROOT / "sql" / "02_modeling_dataset_derivation.sql").read_text(encoding="utf-8")
    for feature in FEATURES:
        assert feature in sql
    assert "is_high_certainty_cohort" in sql
    assert "o.itemid IN (227488, 227489)" in sql
    assert "data/primary_model_dataset.csv" in sql
    assert "data/high_certainty_model_dataset.csv" in sql
    assert "uo_mlkghr_6hr_last" not in sql and "uo_mlkghr_12hr_last" not in sql


def test_table1_uses_locked_labels_and_excludes_outcome():
    data = artificial_training_data()
    data["first_careunit"] = "Medical Intensive Care Unit (MICU)"
    data.loc[:11, "temporal_split"] = "development_2008_2016"
    data.loc[12:, "temporal_split"] = "temporal_test_2017_2022"
    table = build_table1(data, FEATURES)
    assert "outcome_na_ge_151" not in set(table["variable"])
    assert "Last temperature before T0, °C" in set(table["characteristic"])
    assert "Cleaned 6-h urine output rate before T0, mL/kg/h" in set(table["characteristic"])
    assert "No qualifying respiratory-support record before T0" in set(table["characteristic"])
    assert any(value.startswith("First care unit recorded for the ICU stay:") for value in table["characteristic"])
