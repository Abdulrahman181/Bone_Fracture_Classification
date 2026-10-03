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

    def test_all_splits_share_model_preprocessing_and_fixed_class_mapping(self):
        self.assertRegex(self.code, r"Rescaling\(1\.0\s*/\s*255")
        self.assertIn("CLASS_NAMES", self.code)
        self.assertIn("load_datasets", self.code)
        self.assertIn("tf.keras.utils.set_random_seed(SEED)", self.code)

    def test_training_uses_only_train_and_validation_and_test_is_evaluated_once(self):
        fit_calls = re.findall(r"model\.fit\s*\((.*?)\)", self.code, flags=re.DOTALL)
        self.assertEqual(len(fit_calls), 1)
        self.assertIn('datasets["train"]', fit_calls[0])
        self.assertIn('datasets["val"]', fit_calls[0])
        self.assertNotIn('datasets["test"]', fit_calls[0])
        self.assertEqual(self.code.count('model.evaluate(datasets["test"]'), 1)

    def test_artifacts_have_explicit_schema_and_pickle_is_absent(self):
        self.assertIn("write_model_manifest", self.code)
        self.assertIn("MODEL_FILENAME", self.code)
        self.assertNotIn("pickle", self.code.lower())


if __name__ == "__main__":
    unittest.main()
