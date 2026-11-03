# Credit Risk

Survey-based regression of the `mean_risk` target. The repository contains the original exploratory [notebook](Credit%20Risk%20Regression.ipynb) and a small, reproducible [baseline](credit_risk.py).

## Contents

- [Run the baseline](#run-the-baseline)
- [Original notebook](#original-notebook)
- [Saved notebook results](#saved-notebook-results)
- [Focused checks](#focused-checks)
- [License](#license)

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

The notebook expects `data/survey.csv` and imports additional exploratory tools including MLflow, SHAP, boosting libraries, Optuna, TensorFlow, AutoKeras, Mito, and pandas-profiling. Its saved outputs come from a Python 3.10.9 kernel (per the notebook metadata); `requirements-baseline.txt` only installs the new baseline. The MLflow cells log to a tracking server at `http://127.0.0.1:5000/`, and generated files (the profiling report, the AutoKeras project folder and the saved model) are listed in `.gitignore`.

The historical evaluation needs redesign before its scores are used as evidence:

- Imputation, scaling, feature selection, and PCA occur before cross-validation, allowing validation information into preprocessing.
- The neural-network evaluation reuses fitted model weights across folds.

Several source-level mistakes have since been fixed in the notebook code (selected-feature and PCA column names, full-data folds split on `X_outer_val` but applied to `X`, nested Keras callback lists, the MLflow tracking URI set after the experiment lookup, and an absolute save path). The saved outputs were not regenerated, so they still reflect the original code.

The notebook is retained as the original research record. Use the baseline for a controlled starting point and rerun any revised experiment before reporting new results.

## Saved notebook results

The numbers below are copied from the outputs saved in the notebook (December 2022 run, 91,481 survey rows, target `mean_risk`). They predate the fixes above and are affected by the preprocessing leakage, and the execution counts show the cells ran across several sessions, so the exact feature set behind each score cannot be confirmed. Treat them as a historical record, not a benchmark.

| Notebook step | Model | Folds | Mean MSE | Mean R2 |
| --- | --- | --- | --- | --- |
| Model comparison | LightGBM (default) | 10 | 0.6706 | 0.1600 |
| Model comparison | XGBoost (default) | 10 | 0.6782 | 0.1505 |
| Model comparison | CatBoost (default) | 10 | 0.6665 | 0.1652 |
| Tuned, outer half | LightGBM | 10 | 0.6791 | 0.1595 |
| Tuned, outer half | XGBoost | 10 | 0.6633 | 0.1637 |
| Tuned, outer half | CatBoost | 10 | 0.6713 | 0.1599 |
| Tuned, full data | LightGBM | 10 | 0.6699 | 0.1600 |
| Tuned, full data | XGBoost | 10 | 0.6730 | 0.1561 |
| Tuned, full data | CatBoost | 10 | 0.6646 | 0.1677 |
| Ensemble, full data | Voting (LightGBM + XGBoost + CatBoost) | 10 | 0.6644 | 0.1680 |
| AutoKeras, outer half | Best searched network | 2 | 0.6589 | 0.1694 |
| AutoKeras, full data | Best searched network | 10 | 0.6786 | 0.1493 |

The LightGBM and XGBoost full-data rows and the AutoKeras full-data row were produced by the fold-index bug described above. The notebook prints the AutoKeras rows under an "Ensemble" label.

## Focused checks

```bash
python -m unittest discover -s tests -v
```

The baseline checks use synthetic data to verify fold-local preprocessing, deterministic evaluation, input preservation, target separation, and explicit schema errors. The notebook checks read the notebook's source cells (nothing is executed) to guard the fixes listed above. None of them validate the original survey, reproduce historical scores, or train the notebook's models. CI runs the same command on changes to the baseline, the notebook or the tests.

## License

[MIT](LICENSE).
