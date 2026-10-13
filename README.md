# Bone Fracture Classification

This repository contains a reproducible TensorFlow notebook for **image classification of bone fractures**. It is an educational/research artifact, not a clinical device or a medical decision system.

## Data (not included)

The dataset is intentionally not committed because of size/licensing constraints. Set `BONE_FRACTURE_DATA_DIR` to a directory with this exact layout:

```text
<data-root>/
├── train/<class-name>/*.(jpg|jpeg|png|bmp|gif)
├── val/<class-name>/*.(jpg|jpeg|png|bmp|gif)
└── test/<class-name>/*.(jpg|jpeg|png|bmp|gif)
```

The notebook requires all three splits and requires their class-directory names to match. It checks for empty classes and exact file-path overlap between splits before loading data. A missing or invalid dataset raises a clear error; no sample data or metrics are fabricated. Patient-level leakage cannot be assessed without patient identifiers or acquisition metadata, so those controls remain the dataset owner's responsibility.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export BONE_FRACTURE_DATA_DIR=/absolute/path/to/data
jupyter notebook Bone_Fracture_Classification.ipynb
```

Optional environment variables are `BONE_FRACTURE_SEED` (default `42`), `BONE_FRACTURE_EPOCHS` (default `10`), and `BONE_FRACTURE_OUTPUT_DIR` (default `artifacts/`). Run notebook cells from top to bottom. The test split is evaluated once and is never passed to `model.fit`.

## Static validation

Run the repository check without downloading the dataset or TensorFlow:

```bash
python validate_repo.py
```

This checks Python/Notebook JSON syntax, parses every code cell with the AST, rejects stale notebook outputs, and checks for common hard-coded path and test-fitting mistakes. It does not train a model.

## Limitations and cautions

- Reported metrics depend on the supplied split, class balance, random seed, preprocessing, and library versions; this repository does not claim any metric until the user's dataset is supplied and the notebook is run.
- A directory split is not proof of patient-level independence. Use patient/group-aware splitting when identifiers are available.
- Predictions must not be interpreted as professional advice or used for diagnosis without appropriate clinical validation and governance.
