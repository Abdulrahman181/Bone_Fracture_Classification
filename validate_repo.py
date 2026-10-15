"""Dependency-free static checks for this repository.

This intentionally does not import TensorFlow or access the dataset.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Bone_Fracture_Classification.ipynb"
REQUIRED_FILES = (ROOT / "README.md", ROOT / "requirements.txt", NOTEBOOK)
BANNED_PATTERNS = (
    (re.compile(r"/content/|/kaggle/|[A-Za-z]:\\\\"), "non-portable hard-coded path"),
    (
        re.compile(r"model\.fit\s*\(\s*(?:x_)?test(?:_ds)?\b"),
        "test data passed to model.fit",
    ),
)


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def check_python(path: Path, failures: list[str]) -> None:
    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
    except (OSError, SyntaxError) as exc:
        fail(f"{path.relative_to(ROOT)}: {exc}", failures)


def check_notebook(path: Path, failures: list[str]) -> None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"{path.name}: invalid JSON ({exc})", failures)
        return
    if notebook.get("nbformat", 0) < 4:
        fail(f"{path.name}: unsupported nbformat", failures)
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        fail(f"{path.name}: cells must be a list", failures)
        return
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            fail(f"{path.name} cell {index}: cell must be an object", failures)
            continue
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            fail(f"{path.name} cell {index}: stale outputs are committed", failures)
        if cell.get("execution_count") is not None:
            fail(f"{path.name} cell {index}: execution count is not cleared", failures)
        source_lines = cell.get("source", [])
        if not isinstance(source_lines, list) or not all(
            isinstance(line, str) for line in source_lines
        ):
            fail(f"{path.name} cell {index}: source must be a list of strings", failures)
            continue
        source = "".join(source_lines)
        try:
            ast.parse(source, filename=f"{path.name} cell {index}")
            compile(source, f"{path.name} cell {index}", "exec")
        except (SyntaxError, ValueError, TypeError) as exc:
            fail(f"{path.name} cell {index}: {exc}", failures)
        for pattern, description in BANNED_PATTERNS:
            if pattern.search(source):
                fail(f"{path.name} cell {index}: {description}", failures)


def main() -> int:
    failures: list[str] = []
    for required in REQUIRED_FILES:
        if not required.is_file():
            fail(f"missing required file: {required.name}", failures)
    for path in sorted(ROOT.rglob("*.py")):
        if ".git" not in path.parts and ".venv" not in path.parts:
            check_python(path, failures)
    notebooks = sorted(ROOT.rglob("*.ipynb"))
    for notebook_path in notebooks:
        if ".git" not in notebook_path.parts and ".venv" not in notebook_path.parts:
            check_notebook(notebook_path, failures)
            try:
                notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
                source = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in notebook.get("cells", [])
                    if isinstance(cell, dict) and cell.get("cell_type") == "code"
                )
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            for pattern, description in BANNED_PATTERNS:
                if pattern.search(source):
                    fail(f"{notebook_path.relative_to(ROOT)}: {description}", failures)
    if failures:
        print("STATIC VALIDATION FAILED")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("STATIC VALIDATION PASSED: Python and Notebook AST/JSON checks are clean.")
    print("Dataset access and model training were intentionally not attempted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
