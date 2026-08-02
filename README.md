# MIMIC-IV Hypernatremia Landmark Prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21754482.svg)](https://doi.org/10.5281/zenodo.21754482)

Reproducible SQL and Python code for development and temporal validation of a 24-h landmark prediction model for **recorded moderate-to-severe hypernatremia** in MIMIC-IV v3.1.

## Scope

T0 is 24 h after ICU admission. The primary model uses 27 predictors available before T0 and estimates the risk of a first recorded serum sodium value >=151 mmol/L in `[T0, min(ICU admission + 7 days, hospital discharge, death))`. ICU discharge does not end follow-up. The development period was 2008-2016 and temporal validation period was 2017-2022. This is research code, not a clinical decision-support tool.

## Data access and restrictions

This repository contains no MIMIC-IV data, patient-level extracts, predictions, identifiers, timestamps, database credentials, or serialized fitted model objects. Users must obtain credentialed MIMIC-IV access through PhysioNet and comply with the applicable data-use agreement. The authors cannot redistribute patient-level MIMIC-IV data.

## Repository structure

- `sql/`: PostgreSQL cohort, feature, and outcome derivation.
- `scripts/`: model development, calibration, DCA, sensitivity, reporting, and model-specification export scripts.
- `config/`: data-free study and model settings; use `database.example.env` as a template.
- `model_specification/`: frozen coefficients, feature order, and preprocessing parameters.
- `synthetic_example/`: entirely artificial schema/smoke-test data.
- `docs/`: methods, reproduction, limitations, and manuscript-result crosswalk.

## Environment

The public-release verification environment was Python 3.14.3, NumPy 2.4.2, pandas 2.3.3, SciPy 1.17.1, matplotlib 3.10.8, and scikit-learn 1.8.0. The frozen model specification records scikit-learn 1.8.0. The complete original execution environment was not preserved in the locked run manifest; the supplied synthetic smoke test passed in a clean verification environment.

## Reproduction order

1. Configure a local PostgreSQL connection using `config/database.example.env`.
2. Run `sql/01_cohort_derivation.sql` and `sql/02_modeling_dataset_derivation.sql` against MIMIC-IV v3.1 plus required derived concepts.
3. Provide the resulting authorized local modeling data only outside this repository.
4. Run the scripts in `scripts/` for development, temporal validation, calibration, DCA, secondary analysis, and reporting.
5. Compare final implementation parameters against `model_specification/`.

## Limitations

Temporal validation is not independent external validation. The outcome is a recorded sodium event and depends on post-T0 sodium testing. Patients without post-T0 sodium testing were not encoded as non-events. This code must not be used to make unvalidated clinical treatment decisions.

## Citation

Software citation metadata are supplied in `CITATION.cff`. The fixed archived release is [Zenodo version 1.0.1](https://doi.org/10.5281/zenodo.21754482); the all-versions concept DOI is `10.5281/zenodo.21754481`. The public source repository is [gongxun206-rgb/mimic-iv-hypernatremia-landmark-prediction](https://github.com/gongxun206-rgb/mimic-iv-hypernatremia-landmark-prediction). Cite the MIMIC-IV data resource and its applicable access terms separately.
