"""Deterministic serialization boundaries used by every public report."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from mechaudit._serialization import json_safe


class _Status(Enum):
    READY = "ready"


@dataclass(frozen=True)
class _Record:
    value: float


class _DictionaryExport:
    def to_dict(self) -> dict[str, int]:
        return {"answer": 42}


class _StructuredLabel:
    def __repr__(self) -> str:
        return "StructuredLabel('x')"


def test_json_safe_covers_supported_scalars_containers_and_labels(tmp_path: Path) -> None:
    value = {
        "missing": None,
        "finite": 1.25,
        "nan": float("nan"),
        "positive_infinity": float("inf"),
        "negative_infinity": float("-inf"),
        "numpy_scalar": np.int64(3),
        "array": np.array([1.0, float("nan")]),
        "enum": _Status.READY,
        "path": tmp_path / "report.json",
        "mapping": {1: True},
        "set": frozenset({"b", "a"}),
        "sequence": (1, "two"),
        "dataclass": _Record(2.5),
        "exportable": _DictionaryExport(),
        "label": _StructuredLabel(),
    }

    result = json_safe(value)

    assert result == {
        "missing": None,
        "finite": 1.25,
        "nan": None,
        "positive_infinity": "Infinity",
        "negative_infinity": "-Infinity",
        "numpy_scalar": 3,
        "array": [1.0, None],
        "enum": "ready",
        "path": str(tmp_path / "report.json"),
        "mapping": {"1": True},
        "set": ["a", "b"],
        "sequence": [1, "two"],
        "dataclass": {"value": 2.5},
        "exportable": {"answer": 42},
        "label": {"__label_repr__": "StructuredLabel('x')"},
    }

    with pytest.raises(TypeError, match="complex values are not supported"):
        json_safe(1 + 2j)
