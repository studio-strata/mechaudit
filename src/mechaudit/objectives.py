"""Typed linear designer-objective declarations.

Every objective is expressed in the canonical maximization direction. Compiler
backends are responsible for expanding expected objectives using the joint
prior and for preserving objective units in reports.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from mechaudit.core.labels import AgentId, freeze_label
from mechaudit.exceptions import ValidationError
from mechaudit.requirements.base import LinearExpression, VariableCoordinate

AgentWeightsLike = Mapping[Any, float] | Iterable[tuple[Any, float]]


def _weight(value: Any, *, name: str = "weight") -> float:
    if not isinstance(value, Real):
        raise ValidationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return result


def _weights(
    values: AgentWeightsLike | None,
    *,
    required: bool = False,
) -> tuple[tuple[AgentId, float], ...] | None:
    if values is None:
        if required:
            raise ValidationError("agent weights are required")
        return None
    entries = []
    seen = set()
    raw_entries = values.items() if isinstance(values, Mapping) else values
    for raw_agent, raw_weight in raw_entries:
        agent = freeze_label(raw_agent)
        if agent in seen:
            raise ValidationError("agent weights contain duplicate labels")
        seen.add(agent)
        entries.append((agent, _weight(raw_weight)))
    if required and not entries:
        raise ValidationError("agent weights cannot be empty")
    return tuple(entries)


class LinearObjective:
    """Marker base class for immutable linear maximization objectives."""

    @property
    def name(self) -> str:
        """Stable objective name used in reports and fingerprints."""

        return type(self).__name__

    def weighted(self, coefficient: float) -> LinearCombination:
        """Return a scalar-weighted objective combination."""

        return LinearCombination(((coefficient, self),))

    def __mul__(self, coefficient: float) -> LinearCombination:
        return self.weighted(coefficient)

    def __rmul__(self, coefficient: float) -> LinearCombination:
        return self.weighted(coefficient)

    def __add__(self, other: LinearObjective) -> LinearCombination:
        if not isinstance(other, LinearObjective):
            return NotImplemented
        return LinearCombination(((1.0, self), (1.0, other)))


@dataclass(frozen=True)
class ExpectedRevenue(LinearObjective):
    """Expected payments to the designer under truthful reporting."""

    agent_weights: AgentWeightsLike | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_weights", _weights(self.agent_weights))

    @property
    def name(self) -> str:
        return "expected_revenue"


@dataclass(frozen=True)
class ExpectedSocialWelfare(LinearObjective):
    """Expected weighted total value of the selected social outcome."""

    agent_weights: AgentWeightsLike | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_weights", _weights(self.agent_weights))

    @property
    def name(self) -> str:
        return "expected_social_welfare"


@dataclass(frozen=True)
class ExpectedResidualSurplus(LinearObjective):
    """Expected total agent utility: welfare minus designer revenue.

    This explicit definition avoids the conflicting uses of "residual surplus"
    in the literature. It uses the package sign convention ``u = v - p`` and
    weights every agent equally; use :class:`WeightedAgentUtilities` for a
    weighted sum of agent utilities.
    """

    @property
    def name(self) -> str:
        return "expected_residual_surplus"


@dataclass(frozen=True)
class WeightedAgentUtilities(LinearObjective):
    """Expected weighted sum of truthful quasi-linear agent utilities."""

    weights: AgentWeightsLike

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _weights(self.weights, required=True))

    @property
    def name(self) -> str:
        return "weighted_agent_utilities"


@dataclass(frozen=True, init=False)
class LinearCombination(LinearObjective):
    """A weighted combination of other linear objectives."""

    terms: tuple[tuple[float, LinearObjective], ...]
    constant: float

    def __init__(
        self,
        terms: Iterable[tuple[float, LinearObjective]],
        *,
        constant: float = 0.0,
    ) -> None:
        normalized = []
        for raw_coefficient, objective in terms:
            if not isinstance(objective, LinearObjective):
                raise ValidationError("objective combinations require LinearObjective terms")
            coefficient = _weight(raw_coefficient, name="coefficient")
            if coefficient != 0.0:
                normalized.append((coefficient, objective))
        if not normalized:
            raise ValidationError("an objective combination cannot be empty")
        object.__setattr__(self, "terms", tuple(normalized))
        object.__setattr__(self, "constant", _weight(constant, name="constant"))

    @property
    def name(self) -> str:
        return "linear_combination"


@dataclass(frozen=True, init=False)
class CustomLinearObjective(LinearObjective):
    """A labeled affine function of typed allocation/payment coordinates."""

    objective_name: str
    expression: LinearExpression

    def __init__(
        self,
        name: str,
        coefficients: Mapping[VariableCoordinate, float]
        | Iterable[tuple[VariableCoordinate, float]],
        *,
        constant: float = 0.0,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("custom objective name cannot be empty")
        object.__setattr__(self, "objective_name", name.strip())
        object.__setattr__(
            self,
            "expression",
            LinearExpression(coefficients, constant=constant),
        )

    @property
    def name(self) -> str:
        return self.objective_name
