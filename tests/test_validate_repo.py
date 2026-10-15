from __future__ import annotations

import json
from pathlib import Path

from validate_repo import NOTEBOOK, check_notebook


def test_repository_notebook_is_clean() -> None:
    failures: list[str] = []
    check_notebook(NOTEBOOK, failures)
    assert failures == []


def test_notebook_check_rejects_execution_state_and_test_fit(tmp_path: Path) -> None:
    notebook_path = tmp_path / "bad.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 7,
                        "outputs": [],
                        "source": ["model.fit(test_ds)\n"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    failures: list[str] = []
    check_notebook(notebook_path, failures)

    assert any("execution count" in failure for failure in failures)
    assert any("test data passed" in failure for failure in failures)
