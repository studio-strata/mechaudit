"""Typed quasi-linear valuation models."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, Protocol, runtime_checkable

from mechaudit.core.labels import AgentId, OutcomeValue, TypeProfile, TypeValue
from mechaudit.exceptions import EnvironmentValidationError


class ValuationKind(str, Enum):
    """Information dependence declared by a valuation model."""

    PRIVATE = "private"
    INTERDEPENDENT = "interdependent"


@runtime_checkable
class ValuationModel(Protocol):
    """Protocol implemented by all environment valuation providers."""

    @property
    def kind(self) -> ValuationKind:
        """Return the model's declared information dependence."""

    def value(self, agent: AgentId, true_types: TypeProfile, outcome: OutcomeValue) -> float:
        """Return an agent's value before transfers."""


PrivateValueFunction = Callable[[AgentId, TypeValue, OutcomeValue], Real]
InterdependentValueFunction = Callable[[AgentId, TypeProfile, OutcomeValue], Real]


def _finite_value(value: Any, *, context: str) -> float:
    if not isinstance(value, Real):
        raise EnvironmentValidationError(
            "valuation must return a real number",
            path=context,
            context={"value_type": type(value).__name__},
        )
    result = float(value)
    if not math.isfinite(result):
        raise EnvironmentValidationError(
            "valuation must be finite", path=context, context={"value": result}
        )
    return result


@dataclass(frozen=True)
class PrivateValues:
    """Wrap a private-value function with an unambiguous signature.

    The wrapped callback receives only the evaluated agent's own type. This
    makes it impossible for a declared private-value model to accidentally use
    another agent's type.
    """

    function: PrivateValueFunction
    kind: ValuationKind = ValuationKind.PRIVATE

    def value(self, agent: AgentId, true_types: TypeProfile, outcome: OutcomeValue) -> float:
        try:
            own_type = true_types[agent]
        except KeyError as exc:
            raise EnvironmentValidationError(
                "valuation requested an agent absent from the type profile",
                context={"agent": agent},
            ) from exc
        return _finite_value(
            self.function(agent, own_type, outcome),
            context=f"values[{agent!r}, {own_type!r}, {outcome!r}]",
        )

    @classmethod
    def from_mapping(
        cls, values: Mapping[AgentId, Mapping[TypeValue, Mapping[OutcomeValue, Real]]]
    ) -> PrivateValues:
        """Create a private valuation provider from nested labeled mappings."""

        def lookup(agent: AgentId, type_value: TypeValue, outcome: OutcomeValue) -> Real:
            try:
                return values[agent][type_value][outcome]
            except KeyError as exc:
                raise EnvironmentValidationError(
                    "private valuation table is missing an entry",
                    context={"agent": agent, "type": type_value, "outcome": outcome},
                ) from exc

        return cls(lookup)


@dataclass(frozen=True)
class InterdependentValues:
    """Wrap a valuation function that may inspect the complete true profile."""

    function: InterdependentValueFunction
    kind: ValuationKind = ValuationKind.INTERDEPENDENT

    def value(self, agent: AgentId, true_types: TypeProfile, outcome: OutcomeValue) -> float:
        return _finite_value(
            self.function(agent, true_types, outcome),
            context=f"values[{agent!r}, {true_types!r}, {outcome!r}]",
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[AgentId, Mapping[TypeProfile, Mapping[OutcomeValue, Real]]],
    ) -> InterdependentValues:
        """Create an interdependent provider from complete-profile tables."""

        def lookup(agent: AgentId, profile: TypeProfile, outcome: OutcomeValue) -> Real:
            try:
                return values[agent][profile][outcome]
            except KeyError as exc:
                raise EnvironmentValidationError(
                    "interdependent valuation table is missing an entry",
                    context={"agent": agent, "profile": profile, "outcome": outcome},
                ) from exc

        return cls(lookup)
