# Engineering Review — Bone Fracture Classification

**Repository:** `Abdulrahman181/Bone_Fracture_Classification`  
**Review scope:** Python files, Jupyter Notebook, README, dependency and ignore files  
**Review date:** 2026-10-13  
**Execution policy:** local changes only; no GitHub push was performed

## Executive summary

The repository contained one notebook and minimal project metadata. The original notebook was syntactically parseable, but it was not a reliable end-to-end experiment: it used a hard-coded two-class mapping, depended on execution order and local pickles, did not validate the dataset layout, normalized only the training array, used an unshuffled/uncontrolled file traversal, saved a legacy HDF5 model without a controlled output directory, and finally called `model.fit(x_test, y_test)`, which trains on the held-out test set and invalidates the reported evaluation. It also had no static validation or clear missing-data failure mode.

The notebook was replaced with a compact, deterministic TensorFlow workflow that consumes explicit `train/`, `val/`, and `test/` directories, validates class consistency and exact file-path overlap, uses one label mapping for every split, performs preprocessing inside the model for all splits, trains only on the training set, selects using validation data, and evaluates the test split once. Existing notebook outputs and stale Colab metadata were removed. No metric or dataset result was invented.

## Findings and remediation

### Critical / high risk

1. **Test-set leakage:** the original last cell executed `model.fit(x_test, y_test)`. This was removed. The revised notebook calls `model.fit(train_ds, validation_data=val_ds, ...)` only and calls `model.evaluate(test_ds)` once after training.
2. **Inconsistent preprocessing:** the original rescaled `x_train` but evaluated an unscaled `x_test`. A `Rescaling(1 / 255)` layer now sits in the model, so train, validation, and test tensors use the same preprocessing.
3. **Unstable and implicit label mapping:** the original hard-coded `{'not fractured': 0, 'fractured': 1}` and assumed exactly two classes. The revised workflow derives a sorted class list from `train/`, requires identical class directories in `val/` and `test/`, passes that same list to every TensorFlow loader, and sizes the output layer from the observed class count.
4. **Missing data/schema validation:** the old notebook failed with opaque `FileNotFoundError`/`KeyError` errors or could silently skip unreadable images. The revised `validate_dataset` reports the exact missing split, requires at least two classes, rejects empty classes, verifies matching class directories, and rejects exact file-path overlap across splits.
5. **Non-portable and fragile artifacts:** the original used string path concatenation, local pickle files, uncontrolled `os.listdir` traversal, and a working-directory model output. The revised code uses `pathlib`, deterministic sorted traversal for validation, `image_dataset_from_directory`, and an ignored configurable `BONE_FRACTURE_OUTPUT_DIR` with `.keras` checkpoints.

### Medium / maintainability

- Removed unused `opencv-python`, `pandas`, and `tqdm` imports/dependencies; retained only packages used by the revised notebook.
- Added reproducibility controls via `BONE_FRACTURE_SEED`, NumPy/Python/TensorFlow seeding, and TensorFlow deterministic operations where supported.
- Added validation/test metrics using `classification_report` and `confusion_matrix`, without claiming any values before a real dataset run.
- Added `validate_repo.py`, a dependency-free AST/JSON checker that rejects syntax errors, stale outputs, common hard-coded cloud paths, and fitting on test data.
- Expanded README with exact data layout, run instructions, environment variables, validation command, and leakage/clinical-use cautions.
- Expanded `.gitignore` for caches, artifacts, plots, and generated CSV files.

## Data, leakage, and reproducibility assessment

The repository does **not** contain a dataset, and no `BONE_FRACTURE_DATA_DIR` was configured in this environment. The notebook therefore was not executed for training or evaluation. The revised checks can detect exact path overlap when the user's dataset is supplied, but they cannot establish patient-level independence or detect visually duplicated files without patient IDs/acquisition metadata. A production or research-grade run should perform a patient/group-aware split when such identifiers exist and should record the dataset version and class counts.

The class order is deterministic (alphabetical), the random seed is configurable and defaults to `42`, and validation/test datasets are not shuffled. These controls improve repeatability but do not guarantee identical floating-point results across hardware and TensorFlow builds.

## Files changed

- `Bone_Fracture_Classification.ipynb` — rewritten into a validated, reproducible train/validation/test workflow; outputs cleared.
- `README.md` — documented data contract, execution, validation, limitations, and no-fabricated-results policy.
- `requirements.txt` — aligned dependencies and bounded major-version compatibility.
- `requirements-dev.txt` — declared optional pytest and Ruff tooling.
- `tests/test_validate_repo.py` — added regression coverage for notebook hygiene and leakage-pattern checks.
- `.gitignore` — added local artifacts, caches, plots, and generated CSV exclusions.
- `validate_repo.py` — added dependency-free static repository validation.
- `ENGINEERING_REVIEW.md` — this review report.

## Validation performed

- `python3 validate_repo.py` — **passed**; Python and Notebook AST/JSON checks clean, no stale outputs, and no banned test-fitting/path patterns.
- `ruff check . --output-format concise` — **passed** after fixing import ordering in the Notebook.
- `git diff --check` — **passed**.
- Notebook JSON parse — **passed**; `nbformat=4`, 10 cells, 9 code cells, 0 outputs.
- Environment check — TensorFlow unavailable, dataset directory absent; therefore no full notebook execution, model training, or metric generation was claimed.

## Remaining caveats

- Installing TensorFlow and supplying a licensed dataset are prerequisites for an end-to-end run.
- The dependency bounds are intended for a normal Python 3 environment, but exact lockfiles are not added because the repository has no tested runtime matrix.
- The model is a small baseline CNN, not a clinically validated model; threshold calibration, subgroup analysis, external validation, and governance remain out of scope.

## Second Senior Review

**Review date:** 2026-10-15

### Findings

The first review addressed the original test-set fitting, inconsistent preprocessing, implicit class mapping, missing split validation, stale notebook outputs, and non-portable paths. This second pass re-read every tracked file, inspected the complete Git history and prior diffs, and rechecked every notebook/code cell. The remaining material issues were:

1. Dataset validation detected exact path reuse but not duplicated image content copied to different paths across splits, leaving a common form of file-level leakage undetected.
2. The notebook saved a model checkpoint but did not verify that the serialized artifact could be loaded before reporting test results, and it did not persist the class order/preprocessing contract needed to interpret the artifact outside the notebook.
3. Positive integer environment settings were not validated early, so values such as `0`, negatives, or non-integers produced less actionable downstream errors.
4. The repository validator checked only the named notebook and did not reject non-null execution counts or malformed cell source structures; there were no regression tests for those checks.

### Remediation

- Added SHA-256 content-overlap detection between dataset splits while retaining exact resolved-path checks.
- Added positive integer parsing for seed, batch size, and epoch settings with clear errors.
- Added metadata JSON output containing class order, image size, seed, and batch size; the notebook now reloads the saved `.keras` artifact before test evaluation and writes the actual test report/metrics only after a real run.
- Hardened `validate_repo.py` to inspect every notebook, validate notebook cell structures, reject stale execution counts, and detect `model.fit(test_ds)`-style patterns.
- Added `tests/test_validate_repo.py` and `requirements-dev.txt`, plus README instructions for the test/lint workflow.
- No data, metrics, or synthetic model outputs were added. No end-to-end training claim is made because this checkout has no dataset and TensorFlow is not installed.

### Second-pass validation and limitations

- `python3 validate_repo.py` passed.
- `python3 -m compileall -q .` passed.
- `ruff check . --output-format concise` passed.
- `python3 -m pytest -q` passed for the new dependency-free regression tests.
- `python3 -m pip install --dry-run --ignore-installed -r requirements.txt` resolved successfully to a compatible TensorFlow 2.18.1 dependency set without installing packages.
- `git diff --check` passed.
- Notebook JSON contains 10 cells, zero outputs, and null execution counts; all code cells parse and compile.
- TensorFlow and the dataset remain unavailable in this environment, so image decoding, model training, checkpoint serialization, and real metric generation were not executed.
- Content hashes detect byte-identical files, not transformed/visually duplicate images; patient-level independence still requires trusted identifiers and group-aware splitting.
