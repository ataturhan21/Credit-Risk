"""Static regression checks on the notebook source; nothing is executed."""
import json
import re
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "Credit Risk Regression.ipynb"


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


class NotebookSourceTests(unittest.TestCase):
    def setUp(self):
        self.cells = code_cells()
        self.source = "\n".join(self.cells)

    def test_selected_and_pca_columns_are_not_positional_feature_names(self):
        self.assertNotIn("columns=X.columns[:15]", self.source)
        self.assertNotIn("columns=X.columns[:20]", self.source)
        self.assertIn("fs.get_feature_names_out()", self.source)

    def test_folds_are_split_on_the_frame_they_index(self):
        for cell in self.cells:
            if "kf.split(X_outer_val)" in cell:
                self.assertNotIn("X.iloc[train_index]", cell)

    def test_keras_callbacks_are_not_nested(self):
        self.assertNotIn("callbacks=[callback]", self.source)

    def test_tracking_uri_is_set_before_experiment_lookup(self):
        cell = next(c for c in self.cells if "create_experiment" in c)
        self.assertLess(cell.index("set_tracking_uri"), cell.index("create_experiment"))

    def test_no_absolute_user_paths_or_empty_code_cells(self):
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]+Users[\\/]", self.source))
        self.assertTrue(all(cell.strip() for cell in self.cells))


if __name__ == "__main__":
    unittest.main()
