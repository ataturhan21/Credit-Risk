"""Reproducible numeric-survey regression baseline with fold-local preprocessing."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def prepare_data(frame, target="mean_risk", feature_start="maritalstatus", feature_end="katkipayi"):
    """Validate the original notebook's numeric schema without learning statistics."""
    if not frame.columns.is_unique:
        raise ValueError("Input columns must be unique.")
    missing = {target, feature_start, feature_end}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    start = frame.columns.get_loc(feature_start)
    end = frame.columns.get_loc(feature_end)
    if start > end:
        raise ValueError("The feature-start column must precede feature-end.")
    features = frame.iloc[:, start:end + 1].copy()
    if target in features:
        raise ValueError("The target must not appear inside the feature range.")
    try:
        features = features.apply(pd.to_numeric, errors="raise").astype(float)
        labels = pd.to_numeric(frame[target], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Features and target must contain numeric values.") from error
    if np.isinf(features.to_numpy()).any():
        raise ValueError("Features must not contain infinity.")
    if not np.isfinite(labels.to_numpy()).all():
        raise ValueError("Target values must be finite and non-missing; they are never imputed.")
    return features, labels


def build_pipeline():
    """Each CV clone fits its own imputer, scaler, and regressor."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0)),
    ])


def evaluate(frame, folds=5, seed=42, target="mean_risk",
             feature_start="maritalstatus", feature_end="katkipayi"):
    features, labels = prepare_data(frame, target, feature_start, feature_end)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer of at least 2.")
    if len(frame) < 2 * folds:
        raise ValueError("Each validation fold needs at least 2 rows for R2.")
    scores = cross_validate(
        build_pipeline(), features, labels,
        cv=KFold(n_splits=folds, shuffle=True, random_state=seed),
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        n_jobs=1, error_score="raise",
    )
    mse, r2 = -scores["test_mse"], scores["test_r2"]
    return {
        "model": "median imputation + standard scaling + Ridge(alpha=1.0)",
        "rows": len(frame), "features": features.columns.tolist(),
        "target": target, "folds": folds, "seed": seed,
        "mse_mean": float(mse.mean()), "mse_std": float(mse.std()),
        "r2_mean": float(r2.mean()), "r2_std": float(r2.std()),
        "fold_mse": mse.tolist(), "fold_r2": r2.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Local survey CSV; no data is uploaded.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", default="mean_risk")
    parser.add_argument("--feature-start", default="maritalstatus")
    parser.add_argument("--feature-end", default="katkipayi")
    args = parser.parse_args()
    report = evaluate(pd.read_csv(args.csv), args.folds, args.seed, args.target,
                      args.feature_start, args.feature_end)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
