# Credit Risk

Survey-based regression of the `mean_risk` target. The repository contains the original exploratory [notebook](Credit%20Risk%20Regression.ipynb) and a small, reproducible [baseline](credit_risk.py).

## Run the baseline

Use Python 3.12 in a separate environment, install the baseline dependencies, and supply your own survey CSV:

```bash
python -m venv .venv
# Activate .venv for your operating system.
python -m pip install -r requirements-baseline.txt
python credit_risk.py data/survey.csv --folds 5 --seed 42
```

The default schema follows the notebook: numeric feature columns from `maritalstatus` through `katkipayi`, inclusive, and the separate numeric target `mean_risk`. Override these names with `--feature-start`, `--feature-end`, and `--target`. Missing features are imputed; missing or infinite targets are rejected. The source dataset is not included and is never uploaded by this script.

Each cross-validation fold fits its own median imputer, standard scaler, and Ridge regressor on the training rows. The JSON report includes the row count, feature names, seed, fold scores, and mean/standard deviation of MSE and R2. This is a new baseline, not a reproduction of the notebook's LightGBM, XGBoost, CatBoost, or AutoKeras scores. Random folds assume independent survey rows; repeated respondents or time-dependent observations require grouped or chronological splits.

## Original notebook

The notebook expects `data/survey.csv` and imports additional exploratory tools including MLflow, SHAP, boosting libraries, Optuna, TensorFlow, AutoKeras, Mito, and pandas-profiling. Its saved metadata and outputs come from an older environment; `requirements-baseline.txt` only installs the new baseline.

The historical evaluation needs redesign before its scores are used as evidence:

- Imputation, scaling, feature selection, and PCA occur before cross-validation, allowing validation information into preprocessing.
- Some full-data evaluation cells create fold indices from `X_outer_val` but apply them to `X`.
- The neural-network evaluation reuses fitted model weights across folds, and callback lists are nested.
- The MLflow tracking URI is set after experiment lookup; optional paths and packages are environment-specific.

The notebook is retained as the original research record. Use the baseline for a controlled starting point and rerun any revised experiment before reporting new results.

## Focused checks

```bash
python -m unittest discover -s tests -v
```

These checks use synthetic data to verify fold-local preprocessing, deterministic evaluation, input preservation, target separation, and explicit schema errors. They do not validate the original survey, reproduce historical scores, or train the notebook's models.

