"""Structural and executable contracts for the guided Jupyter notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPOSITORY_ROOT / "examples" / "mechaudit_workflow.ipynb"


def _load_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_workflow_notebook_is_clean_and_portable() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert notebook["cells"]
    assert notebook["cells"][0]["cell_type"] == "markdown"

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    assert all("get_ipython(" not in "".join(cell["source"]) for cell in code_cells)


def test_workflow_notebook_code_executes_end_to_end() -> None:
    notebook = _load_notebook()
    namespace: dict[str, Any] = {"__name__": "__main__"}

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compiled = compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")
        exec(compiled, namespace)

    assert namespace["solution"].audit.passed
    assert not namespace["face"].failures
    assert namespace["assessment"].is_feasible
    assert not namespace["assessment"].is_optimal
