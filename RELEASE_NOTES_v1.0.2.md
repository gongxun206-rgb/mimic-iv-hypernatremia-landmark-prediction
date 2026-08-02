# v1.0.2: public reproducibility reconciliation

This release repairs consistency and executability of the public, data-free reproducibility package. It does not change the locked study cohort, outcome labels, model coefficients, performance estimates, or scientific conclusions.

## Changes

- Renamed the at-least-one-measurement population as the primary cohort and the at-least-two-measurement population as the high-certainty cohort.
- Aligned SQL exports with `primary_model_dataset.csv` and `high_certainty_model_dataset.csv`.
- Implemented the locked cleaned 6-h and 12-h urine output rate rules, including weight >=20 kg and exclusion of windows containing GU item IDs 227488 or 227489.
- Updated model-development defaults and added explicit, path-restricted `--overwrite` behavior.
- Reworked reporting to use current project-relative inputs, omit the outcome from Table 1, and reproduce the locked cohort flow.
- Added a joblib-free frozen probability reproduction example and strict expected synthetic probabilities.
- Added database-free synthetic end-to-end tests covering 27 raw inputs, 41 transformed inputs, cohort rules, urine cleaning, outcome missingness, and model fitting.
- Updated GitHub Actions to a stable Python 3.12 environment.
- Expanded README commands and separated original locked-analysis software evidence from public verification environments.

## Data and safety

The repository contains no MIMIC-IV patient-level data, real identifiers, credentials, serialized fitted models, or patient-level predictions. Synthetic examples are entirely artificial.
