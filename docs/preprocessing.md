# Preprocessing

Numeric predictors use training-fold-only median imputation, missing indicators where applicable, and standardization. Sex is one-hot encoded. The 11 support/treatment predictors use locked 0/1 definitions and are not standardized. See `model_specification/`.
