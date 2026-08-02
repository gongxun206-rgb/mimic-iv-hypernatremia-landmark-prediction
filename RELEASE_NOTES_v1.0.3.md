# v1.0.3: CI and release-metadata patch

This patch makes no change to the study cohort, outcome, predictor definitions, model code logic, frozen coefficients, performance results, figures, or clinical conclusions.

## Changes

- Repaired the synthetic GitHub Actions workflow using `actions/checkout@v6` and `actions/setup-python@v6`.
- Added read-only workflow permissions, pip caching, and an explicit Python version check under Python 3.12.
- Updated release snapshot metadata to version 1.0.3 without misattributing the v1.0.2 version DOI.

The version-specific Zenodo DOI is recorded after archive publication. The all-versions concept DOI is `10.5281/zenodo.21754481`.
