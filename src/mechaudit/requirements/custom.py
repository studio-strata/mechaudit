"""Backend-independent custom labeled linear requirements."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from mechaudit.core.labels import freeze_label
from mechaudit.exceptions import RequirementValidationError
from mechaudit.requirements.base import (
    ConstraintId,
    ConstraintRelation,
    ConstraintSpec,
    InputConstraintSense,
    LinearExpression,
    Requirement,
    VariableCoordinate,
)


@dataclass(frozen=True, init=False)
class CustomLinearConstraint(Requirement):
    """A custom affine constraint with a stable semantic label.

    User-facing ``<=`` constraints are multiplied by ``-1`` at construction.
    The stored :attr:`spec` therefore always uses ``expression >= 0`` for an
    inequality or ``expression == 0`` for an equality.
    """

    name: str
    spec: ConstraintSpec
    _family: str

    def __init__(
        self,
        name: str,
        coefficients: Mapping[VariableCoordinate, float]
        | Iterable[tuple[VariableCoordinate, float]],
        *,
        sense: InputConstraintSense | str = InputConstraintSense.GREATER_EQUAL,
        rhs: float = 0.0,
        constant: float = 0.0,
        family: str = "custom",
        economic_unit: str = "custom",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise RequirementValidationError("custom constraint name cannot be empty")
        if not isinstance(family, str) or not family.strip():
            raise RequirementValidationError("custom constraint family cannot be empty")
        try:
            input_sense = InputConstraintSense(sense)
        except ValueError as exc:
            raise RequirementValidationError("sense must be >=, <=, or ==") from exc

        expression = LinearExpression(coefficients, constant=constant).plus(
            LinearExpression(constant=-float(rhs))
        )
        relation = ConstraintRelation.GREATER_EQUAL
        if input_sense is InputConstraintSense.LESS_EQUAL:
            expression = expression.scaled(-1.0)
        elif input_sense is InputConstraintSense.EQUAL:
            relation = ConstraintRelation.EQUAL

        frozen_metadata = tuple(
            sorted(
                (
                    (key, freeze_label(value, path=f"metadata.{key}"))
                    for key, value in (metadata or {}).items()
                ),
                key=lambda item: item[0],
            )
        )
        canonical_family = family.strip().lower()
        if any(key == "name" for key, _ in frozen_metadata):
            raise RequirementValidationError("metadata key 'name' is reserved")
        constraint_id = ConstraintId(
            canonical_family,
            {**dict(frozen_metadata), "name": name.strip()},
        )
        spec = ConstraintSpec(
            id=constraint_id,
            expression=expression,
            relation=relation,
            economic_unit=economic_unit,
            metadata=frozen_metadata,
        )
        object.__setattr__(self, "name", name.strip())
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "_family", canonical_family)

    @property
    def family(self) -> str:
        return self._family


@dataclass(frozen=True)
class CustomLinearRequirement(Requirement):
    """A named group of custom linear constraints for family ablation."""

    name: str
    constraints: tuple[CustomLinearConstraint, ...]

    def __init__(self, name: str, constraints: Iterable[CustomLinearConstraint]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise RequirementValidationError("custom requirement name cannot be empty")
        rows = tuple(constraints)
        if not rows:
            raise RequirementValidationError(
                "a custom linear requirement must contain at least one constraint"
            )
        if any(not isinstance(row, CustomLinearConstraint) for row in rows):
            raise RequirementValidationError(
                "custom requirement rows must be CustomLinearConstraint instances"
            )
        ids = [row.spec.id for row in rows]
        if len(set(ids)) != len(ids):
            raise RequirementValidationError("custom requirement contains duplicate row ids")
        object.__setattr__(self, "name", name.strip())
        object.__setattr__(self, "constraints", rows)

    @property
    def family(self) -> str:
        return "custom_group"


LabeledLinearConstraint = CustomLinearConstraint
