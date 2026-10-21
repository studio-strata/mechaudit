"""Labeled feasibility, capacity, and symmetry declarations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from mechaudit.core.labels import AgentId, Label, OutcomeValue, freeze_label
from mechaudit.exceptions import RequirementValidationError
from mechaudit.requirements.base import Requirement, Selector, freeze_selector


def _finite(value: Any, *, name: str) -> float:
    if not isinstance(value, Real):
        raise RequirementValidationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise RequirementValidationError(f"{name} must be finite")
    return result


@dataclass(frozen=True, init=False)
class Capacity(Requirement):
    """Bound expected resource use at each selected report profile.

    ``usage[outcome]`` declares the resource consumed by an explicit social
    outcome. For each selected report, the mechanism must satisfy
    ``sum_o usage[o] * x(o | report) <= upper`` and, when supplied, the
    analogous lower bound. Explicit outcome feasibility remains the preferred
    representation for hard indivisibility restrictions.
    """

    resource: Label
    usage: tuple[tuple[OutcomeValue, float], ...]
    upper: float
    lower: float | None
    profiles: Selector

    def __init__(
        self,
        *,
        resource: Any,
        usage: Mapping[Any, Real],
        upper: Real,
        lower: Real | None = None,
        profiles: Iterable[Any] | None = None,
    ) -> None:
        if not usage:
            raise RequirementValidationError("capacity usage mapping cannot be empty")
        entries = tuple(
            (freeze_label(outcome), _finite(amount, name="usage"))
            for outcome, amount in usage.items()
        )
        if len({outcome for outcome, _ in entries}) != len(entries):
            raise RequirementValidationError("capacity usage contains duplicate outcomes")
        upper_value = _finite(upper, name="upper")
        lower_value = None if lower is None else _finite(lower, name="lower")
        if lower_value is not None and lower_value > upper_value:
            raise RequirementValidationError("capacity lower bound exceeds upper bound")
        object.__setattr__(self, "resource", freeze_label(resource))
        object.__setattr__(self, "usage", entries)
        object.__setattr__(self, "upper", upper_value)
        object.__setattr__(self, "lower", lower_value)
        object.__setattr__(self, "profiles", freeze_selector(profiles))

    @property
    def family(self) -> str:
        return "capacity"


@dataclass(frozen=True, init=False)
class SymmetryOperation:
    """One explicit relabeling under which a mechanism must be invariant.

    Outcome and type actions are required explicitly; MechAudit never guesses
    how an identity-bearing structured outcome should transform.
    """

    agents: tuple[tuple[AgentId, AgentId], ...]
    types: tuple[tuple[AgentId, tuple[tuple[Label, Label], ...]], ...]
    outcomes: tuple[tuple[OutcomeValue, OutcomeValue], ...]
    name: str

    def __init__(
        self,
        *,
        agents: Mapping[Any, Any],
        types: Mapping[Any, Mapping[Any, Any]],
        outcomes: Mapping[Any, Any],
        name: str = "symmetry",
    ) -> None:
        if not name.strip():
            raise RequirementValidationError("symmetry operation name cannot be empty")
        object.__setattr__(
            self,
            "agents",
            tuple(
                (freeze_label(source), freeze_label(target)) for source, target in agents.items()
            ),
        )
        object.__setattr__(
            self,
            "types",
            tuple(
                (
                    freeze_label(agent),
                    tuple(
                        (freeze_label(source), freeze_label(target))
                        for source, target in mapping.items()
                    ),
                )
                for agent, mapping in types.items()
            ),
        )
        object.__setattr__(
            self,
            "outcomes",
            tuple(
                (freeze_label(source), freeze_label(target)) for source, target in outcomes.items()
            ),
        )
        object.__setattr__(self, "name", name.strip())


@dataclass(frozen=True)
class Symmetry(Requirement):
    """Require allocation and payment invariance under explicit operations."""

    operations: tuple[SymmetryOperation, ...]

    def __init__(self, *operations: SymmetryOperation) -> None:
        if not operations:
            raise RequirementValidationError("Symmetry requires at least one operation")
        if not all(isinstance(operation, SymmetryOperation) for operation in operations):
            raise RequirementValidationError(
                "all symmetry operations must be SymmetryOperation objects"
            )
        object.__setattr__(self, "operations", tuple(operations))

    @property
    def family(self) -> str:
        return "symmetry"


Anonymity = Symmetry
