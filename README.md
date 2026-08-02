# MIMIC-IV Hypernatremia Landmark Prediction

[![synthetic-ci](https://github.com/gongxun206-rgb/mimic-iv-hypernatremia-landmark-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/gongxun206-rgb/mimic-iv-hypernatremia-landmark-prediction/actions/workflows/ci.yml)

Data-free SQL, model-development code, frozen LASSO parameters, and synthetic tests for a MIMIC-IV v3.1 landmark prediction study. This is research software, not a clinical decision-support tool.

## Locked study contract

- T0: 24 h after ICU admission.
- Primary cohort: at least one pre-T0 sodium measurement and all observed pre-T0 values 135-145 mmol/L.
- High-certainty cohort: at least two pre-T0 sodium measurements and all observed pre-T0 values 135-145 mmol/L.
- Outcome: first recorded serum sodium >=151 mmol/L in `[T0, min(ICU admission + 7 days, hospital discharge, death))`.
- ICU discharge does not end follow-up. No post-T0 sodium measurement is not encoded as a non-event.
- Development: 2008-2016. Temporal validation: 2017-2022.
- Core model: 27 pre-T0 variables. Prespecified extension: five additional sodium trajectory/monitoring variables.

## Data restrictions

No MIMIC-IV data, real identifiers, patient-level predictions, database credentials, or serialized fitted models are included. Obtain credentialed access through PhysioNet and comply with its data-use agreement. Keep all generated files in the ignored `data/` and `outputs/` directories.

## Environment

### Original locked analysis environment

The locked model outputs were generated with NumPy 2.3.5, pandas 3.0.1, and scikit-learn 1.8.0; the original Python interpreter version was not preserved in the locked run manifest. The public code package was verified under Python 3.12 in GitHub Actions, with an additional local smoke test performed under Python 3.14.3.

### Public-release verification environment

GitHub Actions uses stable Python 3.12. The package pins NumPy 2.4.2, pandas 2.3.3, SciPy 1.17.1, matplotlib 3.10.8, and scikit-learn 1.8.0. A separate local verification was also performed under Python 3.14.3; that interpreter was not the documented original analysis interpreter.

Create an environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
```

Alternatively:

```powershell
conda env create -f environment.yml
conda activate mimic-iv-hypernatremia-landmark-prediction
python -m pytest -q
```

## PostgreSQL configuration

Copy the variable names from `config/database.example.env` into your local environment. Do not commit the populated file.

```powershell
$env:DB_HOST = "localhost"
$env:DB_PORT = "5432"
$env:DB_NAME = "mimiciv31"
$env:DB_USER = "your_local_user"
$env:PGPASSWORD = "your_local_password"
```

Run from the repository root so that `\copy` writes into the ignored `data/` directory:

```powershell
psql -h $env:DB_HOST -p $env:DB_PORT -U $env:DB_USER -d $env:DB_NAME -v ON_ERROR_STOP=1 -f sql/01_cohort_derivation.sql
psql -h $env:DB_HOST -p $env:DB_PORT -U $env:DB_USER -d $env:DB_NAME -v ON_ERROR_STOP=1 -f sql/02_modeling_dataset_derivation.sql
```

The second script writes:

- `data/primary_model_dataset.csv`
- `data/high_certainty_model_dataset.csv`

## Model development

Primary analysis:

```powershell
python scripts/model_development.py --input data/primary_model_dataset.csv --output outputs/primary_model --analysis-role primary
```

High-certainty sensitivity analysis:

```powershell
python scripts/model_development.py --input data/high_certainty_model_dataset.csv --output outputs/high_certainty_model --analysis-role high_certainty
```

Existing output directories are rejected. Add `--overwrite` only when intentionally replacing a child directory of `outputs/`.

Prespecified 32-variable models:

```powershell
python scripts/extended_model_analysis.py --input data/primary_model_dataset.csv --output outputs/extended_primary --analysis-role primary
python scripts/extended_model_analysis.py --input data/high_certainty_model_dataset.csv --output outputs/extended_high_certainty --analysis-role high_certainty
python scripts/extended_model_comparison.py
```

## Calibration, decision curves, and reporting

The downstream scripts consume patient-level prediction files created locally by model development; these files remain ignored. Their defaults use `outputs/primary_model`:

```powershell
python scripts/calibration.py
python scripts/decision_curve.py
python scripts/decision_curve_relative_strategy.py
python scripts/reporting.py --input data/primary_model_dataset.csv --output outputs/submission_materials
```

The reporting command verifies the locked counts before producing Figure 1, Table 1, and the missingness table. It does not retrain a model or connect to PostgreSQL.

## Frozen probability reproduction

The frozen 41-input LASSO probability can be reproduced without `.pkl` or `.joblib`:

```powershell
python model_specification/probability_reproduction_example.py
python scripts/frozen_model_predict.py --input synthetic_example/synthetic_example.csv --output outputs/synthetic_probabilities.csv
python scripts/export_model_specification.py --output outputs/model_specification_validation
python scripts/validate_public_contract.py --output outputs/public_contract_validation.json
```

The example checks the complete intercept, feature order, preprocessing parameters, and coefficients against stored expected probabilities at absolute tolerance `1e-12`.

## Tests

```powershell
python -m pytest -q
```

The tests use only entirely artificial rows and cover the 27 raw variables, 41 transformed inputs, frozen probability calculation, a small LASSO fit, cohort eligibility, GU-irrigant urine cleaning, and the rule that absent post-T0 sodium cannot be encoded as a non-event.

## Repository structure

- `sql/`: PostgreSQL cohort, feature, and outcome derivation.
- `scripts/`: model development, calibration, DCA, reporting, and data-free helpers.
- `config/`: locked predictor/settings manifests and data-free flow counts.
- `model_specification/`: frozen coefficients, feature order, preprocessing, and probability example.
- `synthetic_example/`: entirely artificial input rows.
- `tests/`: database-free unit and end-to-end tests.
- `docs/`: methods, reproduction notes, limitations, and manuscript crosswalks.

## Citation

This release is archived through Zenodo. The version-specific DOI is recorded in the GitHub release and Zenodo record after publication. The all-versions concept DOI is [`10.5281/zenodo.21754481`](https://doi.org/10.5281/zenodo.21754481).
