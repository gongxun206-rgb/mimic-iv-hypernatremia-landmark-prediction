"""Validate and copy the already frozen, data-free model specification."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pandas as pd
import yaml


PROJECT = Path(__file__).resolve().parents[1]
SPEC_DIR = PROJECT / "model_specification"


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the public frozen model specification")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs" / "model_specification")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)

    source_names = [
        "final_coefficients.csv", "final_feature_order.csv", "preprocessing_parameters.csv",
        "model_metadata.json", "expected_synthetic_probability.json",
    ]
    sources = [SPEC_DIR / name for name in source_names]
    if not all(path.is_file() for path in sources):
        raise FileNotFoundError("One or more frozen model-specification files are missing")
    order = pd.read_csv(SPEC_DIR / "final_feature_order.csv").sort_values("transform_order")
    coefficients = pd.read_csv(SPEC_DIR / "final_coefficients.csv").sort_values("transform_order")
    metadata = json.loads((SPEC_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    predictors = yaml.safe_load((PROJECT / "config" / "predictors.yaml").read_text(encoding="utf-8"))["core_predictors"]
    checks = {
        "original_predictors": len(predictors) == 27 and len(set(predictors)) == 27,
        "transformed_features": len(order) == 41,
        "feature_order_matches_coefficients": order["transformed_feature"].tolist() == coefficients["transformed_feature"].tolist(),
        "final_C": float(metadata["lasso_final_C"]) == 0.3,
        "intercept": float(metadata["intercept"]) == -3.707283461539647,
        "nonzero_coefficients": int((order["coefficient"].abs() > 1e-10).sum()) == 32,
    }
    if not all(checks.values()):
        raise ValueError(f"Frozen specification validation failed: {checks}")
    for source in sources:
        shutil.copy2(source, output / source.name)
    validation = {
        "status": "PASS", "model_refit": False, "checks": checks,
        "source_hashes": {path.name: file_hash(path) for path in sources},
    }
    (output / "model_specification_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
