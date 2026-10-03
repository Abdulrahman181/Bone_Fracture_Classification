# Bone Fracture Classification

An educational TensorFlow/Keras notebook prototype for binary classification of bone X-ray images. **It is not a medical device and must not be used for diagnosis, triage, treatment, or any clinical decision.** No clinical validation, external validation, dataset provenance, or performance result is provided or implied.

## Repository contents and data

The repository contains the notebook, reusable input-validation utilities, and tests. It does not include a dataset or trained model. Obtain data separately and place it in this exact directory structure (class names and spelling are significant):

```text
Dataset/
├── train/
│   ├── fractured/
│   └── not fractured/
├── val/
│   ├── fractured/
│   └── not fractured/
└── test/
    ├── fractured/
    └── not fractured/
```

Put `.jpg` or `.jpeg` files directly inside each class directory. The loader rejects missing/empty splits, unexpected directory entries, symlinks, malformed images, and byte-identical files shared across splits. It reports aggregate counts only and uses a fixed class mapping (`fractured: 0`, `not fractured: 1`). Images are decoded in batches rather than accumulated as full NumPy arrays.

These checks **do not verify patient/study-level separation**, detect near-duplicates, establish label quality, or validate data provenance, licensing, de-identification, or external validity. Those require independent dataset records and review; exact-file duplicate detection is only a limited guard. Use only data you are authorized to access and process. Do not commit images, credentials, patient information, notebook outputs, or generated model files. This repository has no license file; do not assume redistribution rights for the code or any dataset.

## Environment and run instructions

Pinned runtime dependencies target **Python 3.10 or 3.11**. From the repository root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab Bone_Fracture_Classification.ipynb
```

Run the notebook top to bottom with the environment's Python kernel. By default it reads `Dataset/` relative to the working directory; set `BONE_FRACTURE_DATA` to use another local data root. Set `BONE_FRACTURE_ARTIFACT_DIR` to change the output directory (default `artifacts/`). The notebook does not download data. TensorFlow training may be slow or memory-intensive on CPU; no hardware or runtime benchmark is claimed.

Run tests and Python static compilation with:

```bash
python -m unittest discover -s tests -v
python -m compileall -q bone_fracture tests
```

GitHub Actions installs the pinned requirements and runs these checks on Python 3.10 and 3.11.

## Training and evaluation

The notebook seeds Python/TensorFlow, enables deterministic TensorFlow operations, and uses batched directory datasets. The same model-embedded `/255` rescaling layer processes every split. Only `train/` is used by `model.fit`; `val/` controls early stopping with restoration of the best validation-loss weights; `test/` is evaluated exactly once at the end. Test results are aggregate values for that local run only and are not clinical or external validation.

The model is saved locally as `bone-fracture.keras`, with `bone-fracture.metadata.json` containing a versioned input/class/preprocessing schema and model SHA-256 checksum. Both are ignored by Git. The saved notebook has no cached outputs. No dataset-backed training, test evaluation, or metric reproduction was performed for this repository change.

## Security, privacy, and safety

Input errors avoid including filenames in user-facing validation messages. The notebook displays only aggregate class counts and does not embed data outputs in the committed file. Model sidecar metadata contains no dataset paths, image names, patient records, or sample metrics. Model artifacts are treated as local, untrusted outputs; do not load artifacts obtained from untrusted sources. If a credential was ever shared or exposed outside this repository, revoke it with its provider and rotate any dependent credentials.
