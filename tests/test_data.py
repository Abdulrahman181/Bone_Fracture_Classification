import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from bone_fracture.artifacts import write_model_manifest
from bone_fracture.data import CLASS_NAMES, SPLIT_NAMES, inspect_dataset, load_datasets


def create_dataset(root: Path, *, duplicate_across_splits: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for split_index, split in enumerate(SPLIT_NAMES):
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            pixel = (split_index * 100, class_index * 100, 20)
            if duplicate_across_splits and split_index > 0 and class_index == 0:
                pixel = (0, 0, 20)
            Image.new("RGB", (4, 4), pixel).save(class_dir / "image.jpg", format="JPEG")


class DatasetValidationTests(unittest.TestCase):
    def test_valid_dataset_returns_aggregate_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            create_dataset(root)
            counts = inspect_dataset(root)
        self.assertEqual(set(counts), set(SPLIT_NAMES))
        for per_class in counts.values():
            self.assertEqual(per_class, {name: 1 for name in CLASS_NAMES})

    def test_missing_root_and_incomplete_split_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Dataset root"):
                inspect_dataset(Path(tmp) / "missing")
            root = Path(tmp) / "dataset"
            (root / "train" / CLASS_NAMES[0]).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "train, val, and test"):
                inspect_dataset(root)

    def test_empty_class_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            for split in SPLIT_NAMES:
                for name in CLASS_NAMES:
                    (root / split / name).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "at least one JPEG"):
                inspect_dataset(root)

    def test_undeclared_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            create_dataset(root)
            (root / "test2").mkdir()
            with self.assertRaisesRegex(ValueError, "only train, val, and test"):
                inspect_dataset(root)

    def test_corrupt_image_is_rejected_without_disclosing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "private-patient-dataset"
            create_dataset(root)
            (root / "train" / CLASS_NAMES[0] / "image.jpg").write_bytes(b"not a jpeg")
            with self.assertRaisesRegex(ValueError, "unreadable or invalid JPEG") as error:
                inspect_dataset(root)
            self.assertNotIn(str(root), str(error.exception))

    def test_exact_duplicate_images_across_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            create_dataset(root, duplicate_across_splits=True)
            with self.assertRaisesRegex(ValueError, "duplicate image bytes"):
                inspect_dataset(root)

    def test_symlink_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            create_dataset(root)
            target = root / "train" / CLASS_NAMES[0] / "image.jpg"
            link = target.with_name("linked.jpg")
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                inspect_dataset(root)

    def test_input_dimensions_are_validated_before_tensorflow_import(self):
        with self.assertRaisesRegex(ValueError, "image_size"):
            load_datasets("unused", image_size=(0, 100))
        with self.assertRaisesRegex(ValueError, "batch_size"):
            load_datasets("unused", batch_size=0)


class ModelManifestTests(unittest.TestCase):
    def test_manifest_is_checksummed_versioned_and_privacy_minimal(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "bone-fracture.keras"
            payload = b"synthetic model bytes"
            model_path.write_bytes(payload)
            manifest_path = write_model_manifest(
                model_path, best_epoch=2, epochs_completed=4, seed=42, image_size=(64, 32)
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["model_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(manifest["class_to_index"], {name: index for index, name in enumerate(CLASS_NAMES)})
        self.assertEqual(manifest["input"]["height"], 64)
        self.assertEqual(manifest["input"]["width"], 32)
        self.assertIn("not for clinical use", manifest["intended_use"].lower())
        self.assertNotIn("patient", json.dumps(manifest).lower())
        self.assertNotIn(str(tmp), json.dumps(manifest))

    def test_manifest_rejects_invalid_epoch_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "bone-fracture.keras"
            model_path.write_bytes(b"model")
            with self.assertRaisesRegex(ValueError, "best_epoch"):
                write_model_manifest(model_path, best_epoch=5, epochs_completed=4, seed=42)


if __name__ == "__main__":
    unittest.main()
