"""Privacy-minimal metadata for locally generated model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .data import CLASS_NAMES

MANIFEST_SCHEMA_VERSION = 1
MODEL_FILENAME = "bone-fracture.keras"


def write_model_manifest(
    model_path: str | Path,
    *,
    best_epoch: int,
    epochs_completed: int,
    seed: int,
    image_size: tuple[int, int] = (100, 100),
) -> Path:
    """Write a checksummed, atomic sidecar containing schema/config, not patient data."""
    path = Path(model_path)
    if path.name != MODEL_FILENAME or not path.is_file():
        raise ValueError("Expected an existing bone-fracture.keras model artifact.")
    if type(epochs_completed) is not int or epochs_completed < 1:
        raise ValueError("epochs_completed must be a positive integer.")
    if type(best_epoch) is not int or not 1 <= best_epoch <= epochs_completed:
        raise ValueError("best_epoch must be between 1 and epochs_completed.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if len(image_size) != 2 or any(type(dimension) is not int or dimension < 1 for dimension in image_size):
        raise ValueError("image_size must contain two positive integers.")

    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_file": MODEL_FILENAME,
        "model_sha256": digest.hexdigest(),
        "class_to_index": {name: index for index, name in enumerate(CLASS_NAMES)},
        "input": {"height": image_size[0], "width": image_size[1], "channels": 3, "color_mode": "RGB"},
        "preprocessing": {"rescaling": "divide_by_255", "resize_interpolation": "bilinear"},
        "training": {"seed": seed, "best_epoch_by_validation_loss": best_epoch, "epochs_completed": epochs_completed},
        "intended_use": "Educational research only; not for clinical use.",
    }
    manifest_path = path.with_name("bone-fracture.metadata.json")
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=".bone-fracture.metadata.", suffix=".tmp", delete=False,
        ) as temporary:
            temp_path = temporary.name
            json.dump(manifest, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temp_path, manifest_path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)
    return manifest_path
