"""Participation/individual-rationality requirement declarations."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol, runtime_checkable

from mechaudit.core.labels import AgentId, TypeProfile, TypeValue, freeze_label
from mechaudit.exceptions import RequirementValidationError
from mechaudit.requirements.base import (
    Requirement,
    Selector,
    SelectorLike,
    SelectorMap,
    SelectorMapLike,
    ZeroProbabilityPolicy,
    freeze_selector,
    freeze_selector_map,
)


def _outside_value(value: Any) -> float:
    if not isinstance(value, Real):
        raise RequirementValidationError("outside options must be real numbers")
    result = float(value)
    if not math.isfinite(result):
        raise RequirementValidationError("outside options must be finite")
    return result


@runtime_checkable
class OutsideOption(Protocol):
    """A profile-aware outside-option provider."""

    def value(self, agent: AgentId, true_types: TypeProfile) -> float:
        """Return the agent's outside utility at a true profile."""


@dataclass(frozen=True)
class ConstantOutsideOption:
    """The same outside utility for every agent and profile."""

    amount: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", _outside_value(self.amount))

    def value(self, agent: AgentId, true_types: TypeProfile) -> float:
        return self.amount


@dataclass(frozen=True)
class CallableOutsideOption:
    """An outside option evaluated from an agent and complete true profile."""

    function: Callable[[AgentId, TypeProfile], float]

    def value(self, agent: AgentId, true_types: TypeProfile) -> float:
        return _outside_value(self.function(agent, true_types))


@dataclass(frozen=True, init=False)
class OwnTypeOutsideOptions:
    """Agent- and own-type-specific outside utilities."""

    entries: tuple[tuple[AgentId, tuple[tuple[TypeValue, float], ...]], ...]

    def __init__(self, values: Mapping[Any, Mapping[Any, float]]) -> None:
        entries = []
        seen_agents = set()
        for raw_agent, by_type in values.items():
            agent = freeze_label(raw_agent)
            if agent in seen_agents:
                raise RequirementValidationError("outside-option table contains duplicate agents")
            seen_agents.add(agent)
            type_entries = []
            seen_types = set()
            for raw_type, amount in by_type.items():
                type_value = freeze_label(raw_type)
                if type_value in seen_types:
                    raise RequirementValidationError(
                        "outside-option table contains duplicate types"
                    )
                seen_types.add(type_value)
                type_entries.append((type_value, _outside_value(amount)))
            entries.append((agent, tuple(type_entries)))
        object.__setattr__(self, "entries", tuple(entries))

    def value(self, agent: AgentId, true_types: TypeProfile) -> float:
        own_type = true_types[agent]
        for candidate_agent, type_entries in self.entries:
            if candidate_agent == agent:
                for candidate_type, amount in type_entries:
                    if candidate_type == own_type:
                        return amount
                break
        raise RequirementValidationError(
            "outside-option table is missing an agent/type entry",
            context={"agent": agent, "type": own_type},
        )


@dataclass(frozen=True, init=False)
class ProfileOutsideOptions:
    """Agent- and complete-profile-specific outside utilities."""

    entries: tuple[tuple[AgentId, tuple[tuple[TypeProfile, float], ...]], ...]

    def __init__(self, values: Mapping[Any, Mapping[TypeProfile, float]]) -> None:
        entries = []
        seen_agents = set()
        for raw_agent, by_profile in values.items():
            agent = freeze_label(raw_agent)
            if agent in seen_agents:
                raise RequirementValidationError("outside-option table contains duplicate agents")
            seen_agents.add(agent)
            profile_entries = []
            seen_profiles = set()
            for profile, amount in by_profile.items():
                if not isinstance(profile, TypeProfile):
                    raise RequirementValidationError(
                        "profile outside-option keys must be TypeProfile instances"
                    )
                if profile in seen_profiles:
                    raise RequirementValidationError(
                        "outside-option table contains duplicate profiles"
                    )
                seen_profiles.add(profile)
                profile_entries.append((profile, _outside_value(amount)))
            entries.append((agent, tuple(profile_entries)))
        object.__setattr__(self, "entries", tuple(entries))

    def value(self, agent: AgentId, true_types: TypeProfile) -> float:
        for candidate_agent, profile_entries in self.entries:
            if candidate_agent == agent:
                for profile, amount in profile_entries:
                    if profile == true_types:
                        return amount
                break
        raise RequirementValidationError(
            "outside-option table is missing an agent/profile entry",
            context={"agent": agent, "profile": true_types},
        )


OutsideOptionLike = (
    float
    | OutsideOption
    | Callable[[AgentId, TypeProfile], float]
    | Mapping[Any, Mapping[Any, float]]
)


def coerce_outside_option(value: Any) -> OutsideOption:
    """Normalize supported outside-option input styles."""

    if isinstance(value, Real):
        return ConstantOutsideOption(float(value))
    if isinstance(value, OutsideOption):
        return value
    if isinstance(value, Mapping):
        return OwnTypeOutsideOptions(value)
    if callable(value):
        return CallableOutsideOption(value)
    raise RequirementValidationError(
        "outside_option must be numeric, a provider, a callable, or a nested mapping"
    )


def _zero_policy(value: Any) -> ZeroProbabilityPolicy:
    try:
        return ZeroProbabilityPolicy(value)
    except ValueError as exc:
        raise RequirementValidationError(
            "zero_probability_types must be 'error', 'vacuous', or 'enforce_ex_post'"
        ) from exc


@dataclass(frozen=True, init=False)
class ExPostIR(Requirement):
    """Profile-by-profile individual rationality."""

    outside_option: OutsideOption
    agents: Selector
    types: SelectorMap

    def __init__(
        self,
        outside_option: OutsideOptionLike = 0.0,
        *,
        agents: SelectorLike = None,
        types: SelectorMapLike = None,
    ) -> None:
        object.__setattr__(self, "outside_option", coerce_outside_option(outside_option))
        object.__setattr__(self, "agents", freeze_selector(agents))
        object.__setattr__(self, "types", freeze_selector_map(types))

    @property
    def family(self) -> str:
        return "ex_post_ir"


@dataclass(frozen=True, init=False)
class InterimIR(Requirement):
    """Type-interim individual rationality under the joint prior."""

    outside_option: OutsideOption
    agents: Selector
    types: SelectorMap
    zero_probability_types: ZeroProbabilityPolicy

    def __init__(
        self,
        outside_option: OutsideOptionLike = 0.0,
        *,
        agents: SelectorLike = None,
        types: SelectorMapLike = None,
        zero_probability_types: ZeroProbabilityPolicy | str = ZeroProbabilityPolicy.ERROR,
    ) -> None:
        object.__setattr__(self, "outside_option", coerce_outside_option(outside_option))
        object.__setattr__(self, "agents", freeze_selector(agents))
        object.__setattr__(self, "types", freeze_selector_map(types))
        object.__setattr__(
            self,
            "zero_probability_types",
            _zero_policy(zero_probability_types),
        )

    @property
    def family(self) -> str:
        return "interim_ir"


@dataclass(frozen=True, init=False)
class ExAnteIR(Requirement):
    """Agent-level individual rationality before types are realized."""

    outside_option: OutsideOption
    agents: Selector

    def __init__(
        self,
        outside_option: OutsideOptionLike = 0.0,
        *,
        agents: SelectorLike = None,
    ) -> None:
        object.__setattr__(self, "outside_option", coerce_outside_option(outside_option))
        object.__setattr__(self, "agents", freeze_selector(agents))

    @property
    def family(self) -> str:
        return "ex_ante_ir"
