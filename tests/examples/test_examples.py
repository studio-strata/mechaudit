from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIRECTORY = REPOSITORY_ROOT / "examples"
EXAMPLE_SCRIPTS = tuple(sorted(EXAMPLE_DIRECTORY.glob("*.py")))
OPTIMAL_FACE_IMPLEMENTED = (REPOSITORY_ROOT / "src/mechaudit/optimality/analysis.py").is_file()


def _example_parameters() -> list[object]:
    parameters = []
    for script in EXAMPLE_SCRIPTS:
        marks = []
        if script.name == "revenue_equivalent_optima.py" and not OPTIMAL_FACE_IMPLEMENTED:
            marks.append(
                pytest.mark.xfail(
                    reason="optimal-face analysis is scheduled after the v0.1 solver core",
                    strict=True,
                )
            )
        parameters.append(pytest.param(script, id=script.stem, marks=marks))
    return parameters


def test_example_inventory_is_not_empty() -> None:
    assert EXAMPLE_SCRIPTS, "the documented example inventory unexpectedly disappeared"


@pytest.mark.parametrize("script", _example_parameters())
def test_example_runs_as_a_standalone_script(script: Path) -> None:
    environment = os.environ.copy()
    source_path = str(REPOSITORY_ROOT / "src")
    inherited_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if not inherited_path else source_path + os.pathsep + inherited_path
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, (
        f"{script.name} failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert completed.stdout.strip(), f"{script.name} produced no explanatory output"
    assert "Traceback" not in completed.stderr
