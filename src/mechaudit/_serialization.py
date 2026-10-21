"""Stable JSON and pandas helpers shared by public report objects."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from mechaudit._version import __version__

SCHEMA_VERSION = "1.0"


def json_safe(value: Any) -> Any:
    """Convert supported report values to deterministic JSON-compatible data.

    User callbacks are not serialized. Reports contain evaluated
    numerical data and display labels; they are not executable environment files.
    """

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return value
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [json_safe(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_safe(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if isinstance(value, complex):
        raise TypeError("complex values are not supported in MechAudit JSON reports")
    # Stable labels can be arbitrary hashable research identifiers. Their repr is
    # the safest non-executable fallback and is explicitly tagged as such.
    return {"__label_repr__": repr(value)}


def report_envelope(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a report payload with format and library version metadata."""

    return {
        "schema_version": SCHEMA_VERSION,
        "library_version": __version__,
        "report_kind": kind,
        **{str(key): json_safe(value) for key, value in payload.items()},
    }


def dumps_report(kind: str, payload: Mapping[str, Any], *, indent: int = 2) -> str:
    """Serialize a report payload with deterministic key ordering."""

    return json.dumps(report_envelope(kind, payload), indent=indent, sort_keys=True)


def write_report(
    path: str | Path,
    kind: str,
    payload: Mapping[str, Any],
    *,
    indent: int = 2,
) -> None:
    """Write a UTF-8 JSON report to *path*."""

    Path(path).write_text(dumps_report(kind, payload, indent=indent) + "\n", encoding="utf-8")
