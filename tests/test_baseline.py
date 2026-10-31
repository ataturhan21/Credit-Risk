"""Synthetic-data checks; no original survey or optional notebook packages."""
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from credit_risk import build_pipeline, evaluate, prepare_data


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "maritalstatus": [0, 1] * 10,
            "income": [float(i) if i % 4 else np.nan for i in range(20)],
            "katkipayi": np.arange(20, dtype=float),
            "mean_risk": np.arange(20, dtype=float) * 2 + 1,
        })

    def test_holdout_values_do_not_change_imputation_or_scaling(self):
        features, labels = prepare_data(self.frame)
        train, validation = next(KFold(5, shuffle=True, random_state=42).split(features))
        before = clone(build_pipeline()).fit(features.iloc[train], labels.iloc[train])
        changed = features.copy()
        changed.iloc[validation] = 1e9
        after = clone(build_pipeline()).fit(changed.iloc[train], labels.iloc[train])
        np.testing.assert_array_equal(before["imputer"].statistics_, after["imputer"].statistics_)
        np.testing.assert_array_equal(before["scaler"].mean_, after["scaler"].mean_)

    def test_repeated_evaluation_is_deterministic_and_preserves_input(self):
        original = self.frame.copy(deep=True)
        first = evaluate(self.frame)
        self.assertEqual(first, evaluate(self.frame))
        self.assertEqual(len(first["fold_mse"]), 5)
        self.assertTrue(all(value >= 0 for value in first["fold_mse"]))
        pd.testing.assert_frame_equal(self.frame, original)

    def test_missing_target_is_rejected_instead_of_imputed(self):
        self.frame.loc[0, "mean_risk"] = np.nan
        with self.assertRaisesRegex(ValueError, "never imputed"):
            evaluate(self.frame)

    def test_target_cannot_leak_into_features(self):
        self.frame = self.frame[["maritalstatus", "mean_risk", "income", "katkipayi"]]
        with self.assertRaisesRegex(ValueError, "target must not appear"):
            evaluate(self.frame)

    def test_invalid_schema_and_fold_sizes_are_explained(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            evaluate(self.frame.rename(columns={"katkipayi": "other"}))
        with self.assertRaisesRegex(ValueError, "at least 2 rows"):
            evaluate(self.frame.iloc[:9])
        with self.assertRaisesRegex(ValueError, "precede"):
            prepare_data(self.frame, feature_start="katkipayi", feature_end="maritalstatus")


if __name__ == "__main__":
    unittest.main()
