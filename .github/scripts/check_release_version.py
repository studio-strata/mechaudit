"""Check release metadata against an exact ``vX.Y.Z`` tag."""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
VERSION_PATTERN = re.compile(r"v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)")


def _match_one(pattern: str, text: str, *, source: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"could not read version from {source}")
    return match.group("version")


def validate_version_sources(tag: str, *, root: Path = ROOT) -> str:
    """Validate the tag, Hatch version source, and citation version.

    Return the unprefixed version when every source agrees. Publication metadata
    is validated separately because it is finalized only for a tagged release.
    """

    tag_match = VERSION_PATTERN.fullmatch(tag)
    if tag_match is None:
        raise RuntimeError("release tags must have exact form vX.Y.Z")
    expected = tag_match.group("version")

    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    dynamic = project.get("dynamic", [])
    if "version" in project or not isinstance(dynamic, list) or "version" not in dynamic:
        raise RuntimeError(
            "pyproject.toml must declare project.version as dynamic and must not define a "
            "static project.version"
        )

    try:
        configured_path = pyproject["tool"]["hatch"]["version"]["path"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("pyproject.toml must configure tool.hatch.version.path") from error
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise RuntimeError("tool.hatch.version.path must be a non-empty relative path")

    relative_path = Path(configured_path)
    version_path = (root / relative_path).resolve()
    resolved_root = root.resolve()
    if relative_path.is_absolute() or resolved_root not in version_path.parents:
        raise RuntimeError("tool.hatch.version.path must resolve inside the repository")

    versions = {
        configured_path: _match_one(
            r'^__version__\s*=\s*"(?P<version>[^"]+)"',
            version_path.read_text(encoding="utf-8"),
            source=configured_path,
        ),
        "CITATION.cff": _match_one(
            r"^version:\s*['\"]?(?P<version>[0-9]+\.[0-9]+\.[0-9]+)['\"]?\s*$",
            (root / "CITATION.cff").read_text(encoding="utf-8"),
            source="CITATION.cff",
        ),
    }
    mismatches = {source: version for source, version in versions.items() if version != expected}
    if mismatches:
        details = ", ".join(f"{source}={version}" for source, version in mismatches.items())
        raise RuntimeError(f"release tag {tag} does not match {details}")

    return expected


def validate_publication_metadata(version: str, *, root: Path = ROOT) -> None:
    """Require dated citation and changelog metadata for a tagged release."""

    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    date_match = re.search(
        r"^date-released:\s*(?P<date>\d{4}-\d{2}-\d{2})\s*$", citation, re.MULTILINE
    )
    if date_match is None:
        raise RuntimeError("CITATION.cff must contain date-released for a published release")
    release_date = date_match.group("date")
    try:
        date.fromisoformat(release_date)
    except ValueError as error:
        raise RuntimeError("CITATION.cff date-released must be a valid ISO date") from error

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if (
        re.search(
            rf"^## \[{re.escape(version)}\] - {re.escape(release_date)}$",
            changelog,
            re.MULTILINE,
        )
        is None
    ):
        raise RuntimeError(f"CHANGELOG.md has no [{version}] release section dated {release_date}")


def main(tag: str) -> int:
    """Validate synchronized publication metadata for *tag*."""

    version = validate_version_sources(tag)
    validate_publication_metadata(version)

    print(f"release metadata matches {tag}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_release_version.py vX.Y.Z")
    raise SystemExit(main(sys.argv[1]))
