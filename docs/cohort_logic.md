# Cohort logic

See `sql/01_cohort_derivation.sql`. The primary risk set is restricted to adult first ICU stays alive and hospitalized at T0, with at least one pre-T0 sodium and all measured pre-T0 sodium values 135-145 mmol/L. Pre-T0 3% and 23.4% sodium chloride are excluded. Mannitol is retained as a baseline flag.
