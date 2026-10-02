# Bone Fracture Classification

An educational TensorFlow/Keras notebook prototype for binary classification of bone X-ray images. **It is not a medical device and must not be used for diagnosis, triage, or treatment decisions.** No clinical validation is provided or implied.

## Repository contents and data

The repository contains the notebook only; the dataset and trained model are not included. The notebook expects you to obtain a dataset separately and place it in this layout, with the class directory names shown exactly:

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

Each class directory should contain `.jpg` images. Use only data you are authorized to access and process. The dataset's source, license, patient-level split design, label quality, and de-identification status are not documented here; verify those independently before use. Do not add patient data, credentials, generated models, or dataset contents to Git. The repository has no license file, so do not assume permission to redistribute its code or any data.

## Environment and run instructions

The pinned requirements target **Python 3.10 or 3.11**. From the repository root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab Bone_Fracture_Classification.ipynb
```

Select the environment's Python kernel and run the notebook from top to bottom. The dataset must be present under `Dataset/` relative to the repository root. The notebook does not download data. On a CPU-only computer, TensorFlow training may be slow; no execution time or hardware requirement has been validated here.

Run repository-level static checks without installing the ML dependencies:

```bash
python -m unittest discover -s tests -v
```

## Evaluation boundaries

The notebook applies the same `/255` rescaling layer inside the model to training, validation, and test inputs. Training uses only `train/`; `val/` is passed as `validation_data`; the test split is used only in the final `model.evaluate` call. The notebook saves the locally generated model as `Bone-fracture.keras`, which is ignored by Git.

The saved notebook outputs were cleared, and no new training or evaluation was performed for this change. There are no reproduced performance results. Before treating any future metric as meaningful, establish dataset provenance and licensing, verify de-identification, confirm patient/study-level separation and absence of split leakage, and run the full experiment in a documented environment. Accuracy alone is not evidence of clinical performance.

## Security and privacy

Notebook outputs are intentionally not committed because they included embedded medical images and stale run results. Dataset files and model artifacts are excluded by `.gitignore`. Do not load pickle files from untrusted sources; the notebook no longer writes or loads pickle artifacts. A scan of the current notebook and available repository history found no Kaggle credential or common access-token indicators. This scan cannot establish whether any credential was exposed elsewhere or in an external copy; if you have ever shared a Kaggle token, revoke it in Kaggle and create a replacement yourself.
