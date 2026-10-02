import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "Bone_Fracture_Classification.ipynb"


class NotebookIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        cls.code = "\n".join(cls.code_cells)

    def test_code_cells_are_valid_python(self):
        for index, source in enumerate(self.code_cells):
            with self.subTest(cell=index):
                compile(source, f"notebook-cell-{index}", "exec")

    def test_notebook_has_no_cached_outputs_or_execution_counts(self):
        for index, cell in enumerate(self.notebook["cells"]):
            with self.subTest(cell=index):
                self.assertFalse(cell.get("outputs", []))
                self.assertIsNone(cell.get("execution_count"))

    def test_preprocessing_is_inside_model_for_all_splits(self):
        self.assertRegex(self.code, r"Rescaling\(1\.0\s*/\s*255")
        self.assertIn("x_test = np.asarray(x_test, dtype=np.uint8)", self.code)
        self.assertIn("x_pred = np.asarray(x_pred, dtype=np.uint8)", self.code)

    def test_validation_is_used_and_test_is_not_used_for_training(self):
        fit_calls = re.findall(r"model\.fit\s*\((.*?)\)", self.code, flags=re.DOTALL)
        self.assertEqual(len(fit_calls), 1)
        self.assertIn("x_train", fit_calls[0])
        self.assertRegex(self.code, r"validation_data\s*=\s*\(x_pred,\s*y_true_pred\)")
        self.assertNotIn("x_test", fit_calls[0])
        self.assertNotIn("y_test", fit_calls[0])

    def test_test_split_is_only_evaluated_once_and_pickle_is_absent(self):
        self.assertEqual(self.code.count("model.evaluate(x_test, y_test)"), 1)
        self.assertNotIn("pickle", self.code.lower())


if __name__ == "__main__":
    unittest.main()
