from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/check_release_version.py"
CHECKER: dict[str, Any] = runpy.run_path(str(SCRIPT))


def _metadata_tree(root: Path, *, dynamic: bool = True, citation_version: str = "0.1.0") -> None:
    project_version = 'dynamic = ["version"]' if dynamic else 'version = "0.1.0"'
    (root / "src/mechaudit").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f"""[project]
name = "mechaudit"
{project_version}

[tool.hatch.version]
path = "src/mechaudit/_version.py"
""",
        encoding="utf-8",
    )
    (root / "src/mechaudit/_version.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    (root / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\nversion: {citation_version}\n", encoding="utf-8"
    )


def test_dynamic_hatch_version_and_citation_must_match_tag(tmp_path: Path) -> None:
    _metadata_tree(tmp_path)

    assert CHECKER["validate_version_sources"]("v0.1.0", root=tmp_path) == "0.1.0"


@pytest.mark.parametrize(
    ("dynamic", "citation_version", "message"),
    [
        (False, "0.1.0", "must declare project.version as dynamic"),
        (True, "0.2.0", "does not match"),
    ],
)
def test_invalid_version_metadata_is_rejected(
    tmp_path: Path, dynamic: bool, citation_version: str, message: str
) -> None:
    _metadata_tree(tmp_path, dynamic=dynamic, citation_version=citation_version)

    with pytest.raises(RuntimeError, match=message):
        CHECKER["validate_version_sources"]("v0.1.0", root=tmp_path)


def test_publication_date_is_required_and_must_match_changelog(tmp_path: Path) -> None:
    _metadata_tree(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [0.1.0] - 2026-08-18\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="date-released"):
        CHECKER["validate_publication_metadata"]("0.1.0", root=tmp_path)

    with (tmp_path / "CITATION.cff").open("a", encoding="utf-8") as citation:
        citation.write("date-released: 2026-08-18\n")

    CHECKER["validate_publication_metadata"]("0.1.0", root=tmp_path)
