"""Dataset validation and bounded-memory TensorFlow input pipelines.

The module deliberately validates only file/layout properties. Patient-level split
independence and clinical data quality must be established from dataset provenance.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

CLASS_NAMES = ("fractured", "not fractured")
SPLIT_NAMES = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def inspect_dataset(data_root: str | Path) -> dict[str, dict[str, int]]:
    """Validate a strict train/val/test directory tree and return aggregate counts.

    Images are verified as JPEGs without exposing their names in errors. Exact
    byte-for-byte duplicates across splits are rejected as a basic leakage guard;
    this cannot establish patient/study-level independence or detect near-duplicates.
    """
    root = Path(data_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Dataset root must be an existing, non-symlink directory.")
    root_entries = list(root.iterdir())
    if any(entry.is_symlink() for entry in root_entries):
        raise ValueError("Dataset directories must not contain symlinks.")
    if len(root_entries) != len(SPLIT_NAMES) or {entry.name for entry in root_entries} != set(SPLIT_NAMES):
        raise ValueError("Dataset root must contain only train, val, and test directories.")

    if any((root / split).is_symlink() or not (root / split).is_dir() for split in SPLIT_NAMES):
        raise ValueError("Dataset must contain train, val, and test directories.")
    counts: dict[str, dict[str, int]] = {}
    seen_hashes: dict[str, str] = {}
    for split in SPLIT_NAMES:
        split_dir = root / split
        if split_dir.is_symlink() or not split_dir.is_dir():
            raise ValueError("Dataset must contain train, val, and test directories.")
        children = list(split_dir.iterdir())
        if any(child.is_symlink() for child in children):
            raise ValueError("Dataset directories must not contain symlinks.")
        class_dirs = [child for child in children if child.is_dir()]
        if {child.name for child in class_dirs} != set(CLASS_NAMES) or len(class_dirs) != len(CLASS_NAMES):
            raise ValueError("Each split must contain exactly the documented class directories.")
        if len(children) != len(class_dirs):
            raise ValueError("Split directories may contain only the documented class directories.")

        counts[split] = {}
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            entries = list(class_dir.iterdir())
            if not entries:
                raise ValueError("Every class in every split must contain at least one JPEG image.")
            if any(entry.is_symlink() for entry in entries):
                raise ValueError("Dataset directories must not contain symlinks.")
            if any(not entry.is_file() for entry in entries):
                raise ValueError("Class directories may contain only regular JPEG image files.")
            if any(entry.suffix.lower() not in IMAGE_SUFFIXES for entry in entries):
                raise ValueError("Class directories may contain only .jpg or .jpeg files.")

            for image_path in entries:
                digest = _file_sha256(image_path)
                previous_split = seen_hashes.get(digest)
                if previous_split is not None and previous_split != split:
                    raise ValueError("Exact duplicate image bytes were found across dataset splits.")
                seen_hashes[digest] = split
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", Image.DecompressionBombWarning)
                        with Image.open(image_path) as image:
                            if image.format != "JPEG" or image.width < 1 or image.height < 1:
                                raise ValueError("Image is not a valid JPEG.")
                            image.verify()
                        with Image.open(image_path) as image:
                            image.load()
                except (
                    OSError,
                    ValueError,
                    UnidentifiedImageError,
                    Image.DecompressionBombError,
                    Image.DecompressionBombWarning,
                ):
                    raise ValueError("One or more dataset images are unreadable or invalid JPEGs.") from None
            counts[split][class_name] = len(entries)
    return counts


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise ValueError("One or more dataset images are unreadable.") from None
    return digest.hexdigest()


def load_datasets(
    data_root: str | Path,
    *,
    image_size: tuple[int, int] = (100, 100),
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    """Build deterministic, batched datasets; shuffle only the training split."""
    if len(image_size) != 2 or any(type(dimension) is not int or dimension < 1 for dimension in image_size):
        raise ValueError("image_size must contain two positive integers.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    counts = inspect_dataset(data_root)
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on optional runtime setup
        raise RuntimeError("TensorFlow is required to construct image datasets.") from exc

    datasets: dict[str, Any] = {}
    root = Path(data_root).expanduser()
    for split in SPLIT_NAMES:
        dataset = tf.keras.utils.image_dataset_from_directory(
            root / split,
            labels="inferred",
            label_mode="int",
            class_names=list(CLASS_NAMES),
            color_mode="rgb",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=(split == "train"),
            seed=seed,
            interpolation="bilinear",
            follow_links=False,
        )
        if tuple(dataset.class_names) != CLASS_NAMES:
            raise RuntimeError("TensorFlow returned an unexpected class mapping.")
        options = tf.data.Options()
        options.experimental_deterministic = True
        datasets[split] = dataset.with_options(options).prefetch(tf.data.AUTOTUNE)
    return datasets, counts
