"""Shared requirement declarations and backend-independent linear semantics."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from mechaudit.core.labels import (
    AgentId,
    Label,
    OutcomeValue,
    Profile,
    ReportProfile,
    TypeValue,
    freeze_label,
)
from mechaudit.exceptions import RequirementValidationError


def qualifier_text(value: Any) -> str:
    """Return the display text of one constraint qualifier.

    Type and report profiles are rendered as plain agent-to-type mappings so
    the same constraint receives the same identifier in audit reports, compiled
    LP rows, and optimal-face reports.
    """

    if isinstance(value, Profile):
        return repr(value.to_dict())
    return repr(value)


def format_constraint_id(family: str, qualifiers: Mapping[str, Any]) -> str:
    """Return the canonical semantic identifier shared by audits and compiled rows."""

    if not qualifiers:
        return family
    body = ", ".join(f"{key}={qualifier_text(value)}" for key, value in sorted(qualifiers.items()))
    return f"{family}[{body}]"


class ZeroProbabilityPolicy(str, Enum):
    """Semantics for conditional constraints at zero-marginal types."""

    ERROR = "error"
    VACUOUS = "vacuous"
    ENFORCE_EX_POST = "enforce_ex_post"


class ConstraintRelation(str, Enum):
    """Relations in the canonical economic row representation."""

    GREATER_EQUAL = ">="
    EQUAL = "=="


class InputConstraintSense(str, Enum):
    """Accepted user-facing senses before canonicalization."""

    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="


@dataclass(frozen=True)
class AllocationCoordinate:
    """One allocation-probability variable ``x(outcome | report)``."""

    report: ReportProfile
    outcome: OutcomeValue

    def __post_init__(self) -> None:
        if not isinstance(self.report, ReportProfile):
            raise RequirementValidationError(
                "allocation coordinates require a canonical ReportProfile"
            )
        object.__setattr__(self, "outcome", freeze_label(self.outcome))


@dataclass(frozen=True)
class PaymentCoordinate:
    """One payment variable ``p_agent(report)``.

    Positive values always denote a transfer from the agent to the designer.
    """

    report: ReportProfile
    agent: AgentId

    def __post_init__(self) -> None:
        if not isinstance(self.report, ReportProfile):
            raise RequirementValidationError(
                "payment coordinates require a canonical ReportProfile"
            )
        object.__setattr__(self, "agent", freeze_label(self.agent))


VariableCoordinate = AllocationCoordinate | PaymentCoordinate


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequirementValidationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise RequirementValidationError(f"{name} must be finite")
    return result


@dataclass(frozen=True, init=False)
class LinearExpression:
    """An immutable affine expression over mechanism coordinates."""

    terms: tuple[tuple[VariableCoordinate, float], ...]
    constant: float

    def __init__(
        self,
        terms: Mapping[VariableCoordinate, float] | Iterable[tuple[VariableCoordinate, float]] = (),
        *,
        constant: float = 0.0,
    ) -> None:
        raw_terms = terms.items() if isinstance(terms, Mapping) else terms
        order = []
        combined = {}
        for coordinate, raw_coefficient in raw_terms:
            if not isinstance(coordinate, (AllocationCoordinate, PaymentCoordinate)):
                raise RequirementValidationError(
                    "linear-expression keys must be typed mechanism coordinates"
                )
            coefficient = _finite_number(raw_coefficient, name="coefficient")
            if coordinate not in combined:
                order.append(coordinate)
                combined[coordinate] = 0.0
            combined[coordinate] += coefficient
        normalized = tuple(
            (coordinate, combined[coordinate])
            for coordinate in order
            if combined[coordinate] != 0.0
        )
        object.__setattr__(self, "terms", normalized)
        object.__setattr__(self, "constant", _finite_number(constant, name="constant"))

    @property
    def coefficients(self) -> Mapping[VariableCoordinate, float]:
        """Read-only coordinate-to-coefficient mapping."""

        return MappingProxyType(dict(self.terms))

    def scaled(self, factor: float) -> LinearExpression:
        """Return this expression multiplied by *factor*."""

        scalar = _finite_number(factor, name="factor")
        return LinearExpression(
            ((coordinate, scalar * coefficient) for coordinate, coefficient in self.terms),
            constant=scalar * self.constant,
        )

    def plus(self, other: LinearExpression) -> LinearExpression:
        """Return the sum of two expressions."""

        if not isinstance(other, LinearExpression):
            return NotImplemented
        return LinearExpression(self.terms + other.terms, constant=self.constant + other.constant)


@dataclass(frozen=True, init=False)
class ConstraintId:
    """Stable economic identity independent of backend row positions."""

    family: str
    qualifiers: tuple[tuple[str, Label], ...]

    def __init__(self, family: str, qualifiers: Mapping[str, Any] | None = None) -> None:
        if not isinstance(family, str) or not family.strip():
            raise RequirementValidationError("constraint family must be a nonempty string")
        entries = []
        for key, value in (qualifiers or {}).items():
            if not isinstance(key, str) or not key:
                raise RequirementValidationError("constraint qualifier keys must be strings")
            entries.append((key, freeze_label(value, path=f"constraint_id.{key}")))
        entries.sort(key=lambda item: item[0])
        object.__setattr__(self, "family", family.strip().lower())
        object.__setattr__(self, "qualifiers", tuple(entries))

    def __str__(self) -> str:
        return format_constraint_id(self.family, dict(self.qualifiers))


@dataclass(frozen=True)
class ConstraintSpec:
    """A canonical backend-independent linear row.

    Inequalities always mean ``expression >= 0``. Equalities mean
    ``expression == 0``. This convention gives every inequality the same
    nonnegative-slack definition and keeps backend sign choices internal.
    """

    id: ConstraintId
    expression: LinearExpression
    relation: ConstraintRelation = ConstraintRelation.GREATER_EQUAL
    economic_unit: str = "utility"
    original_scale: float = 1.0
    metadata: tuple[tuple[str, Label], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, ConstraintId):
            raise RequirementValidationError("constraint id must be a ConstraintId")
        if not isinstance(self.expression, LinearExpression):
            raise RequirementValidationError("constraint expression must be a LinearExpression")
        try:
            relation = ConstraintRelation(self.relation)
        except ValueError as exc:
            raise RequirementValidationError("constraint relation must be >= or ==") from exc
        object.__setattr__(self, "relation", relation)
        object.__setattr__(
            self, "original_scale", _finite_number(self.original_scale, name="original_scale")
        )
        if self.original_scale <= 0:
            raise RequirementValidationError("original_scale must be positive")


class Requirement(ABC):
    """Marker interface for immutable economic requirement declarations."""

    @property
    @abstractmethod
    def family(self) -> str:
        """Return the stable semantic constraint-family name."""


Selector = tuple[Label, ...] | None
SelectorMap = tuple[tuple[AgentId, tuple[TypeValue, ...]], ...] | None
SelectorLike = Iterable[Any] | None
SelectorMapLike = Mapping[Any, Iterable[Any]] | Iterable[tuple[Any, Iterable[Any]]] | None


def freeze_selector(values: SelectorLike) -> Selector:
    """Freeze an optional label selector; ``None`` means all, ``()`` none."""

    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        values = (values,)
    frozen = tuple(freeze_label(value) for value in values)
    if len(set(frozen)) != len(frozen):
        raise RequirementValidationError("selector contains duplicate labels")
    return frozen


def freeze_selector_map(
    values: SelectorMapLike,
) -> SelectorMap:
    """Freeze agent-specific type or deviation selectors."""

    if values is None:
        return None
    entries = []
    seen = set()
    raw_entries = values.items() if isinstance(values, Mapping) else values
    for raw_agent, selected in raw_entries:
        agent = freeze_label(raw_agent)
        if agent in seen:
            raise RequirementValidationError("selector map contains duplicate agents")
        seen.add(agent)
        entries.append((agent, freeze_selector(selected) or ()))
    return tuple(entries)
