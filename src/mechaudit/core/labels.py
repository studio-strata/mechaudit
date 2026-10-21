"""Immutable labels and profiles for finite mechanism-design domains.

User-facing labels may be ordinary hashable objects or JSON-like mappings.
Mappings and nested sequences are frozen recursively so they can safely be
used as dictionary keys without losing their readable, labeled structure.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from mechaudit.exceptions import LabelValidationError

Label = Hashable
AgentId = Label
TypeValue = Label
OutcomeValue = Label
LabelLike = Hashable | Mapping[str, Any] | Sequence[Any]


@dataclass(frozen=True)
class StructuredLabel(Mapping[str, Label]):
    """A hashable, immutable mapping used for multidimensional labels.

    Keys are sorted so mapping insertion order does not affect equality,
    hashing, fingerprints, or exported constraint identifiers.
    """

    _items: tuple[tuple[str, Label], ...]

    def __init__(self, values: Mapping[str, Any]) -> None:
        items = []
        for key, value in values.items():
            if not isinstance(key, str):
                raise LabelValidationError(
                    "structured label keys must be strings",
                    context={"key": key},
                )
            items.append((key, freeze_label(value, path=key)))
        items.sort(key=lambda item: item[0])
        object.__setattr__(self, "_items", tuple(items))

    def __getitem__(self, key: str) -> Label:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self._items)
        return f"StructuredLabel({body})"

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable, recursively plain representation."""

        return {key: thaw_label(value) for key, value in self._items}


def freeze_label(value: Any, *, path: str | None = None) -> Label:
    """Convert *value* to an immutable, hashable label.

    Mappings become :class:`StructuredLabel` instances and lists/tuples become
    recursively frozen tuples. Sets are rejected because their iteration order
    is not a stable declaration of a finite domain.
    """

    if isinstance(value, StructuredLabel):
        return value
    if isinstance(value, Mapping):
        return StructuredLabel(value)
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_label(item, path="{}[{}]".format(path or "label", index))
            for index, item in enumerate(value)
        )
    if isinstance(value, (set, frozenset)):
        raise LabelValidationError(
            "sets are not valid labels because their ordering is ambiguous",
            path=path,
        )
    if isinstance(value, float) and math.isnan(value):
        raise LabelValidationError("NaN is not a valid label", path=path)
    if not isinstance(value, Hashable):
        raise LabelValidationError(
            "label must be hashable or a JSON-like mapping/sequence",
            path=path,
            context={"label_type": type(value).__name__},
        )
    try:
        hash(value)
    except (TypeError, ValueError) as exc:
        raise LabelValidationError("label is not safely hashable", path=path) from exc
    return value


def thaw_label(value: Label) -> Any:
    """Convert built-in frozen label containers to plain Python containers."""

    if isinstance(value, StructuredLabel):
        return value.to_dict()
    if isinstance(value, tuple):
        return [thaw_label(item) for item in value]
    return value


def normalize_axis(values: Iterable[Any], *, name: str) -> tuple[Label, ...]:
    """Freeze and validate an explicitly ordered finite-domain axis."""

    if isinstance(values, (set, frozenset)):
        raise LabelValidationError(f"{name} must be an ordered iterable, not a set", path=name)
    frozen = tuple(
        freeze_label(value, path=f"{name}[{index}]") for index, value in enumerate(values)
    )
    if not frozen:
        raise LabelValidationError(f"{name} cannot be empty", path=name)

    seen: dict[Label, int] = {}
    for index, value in enumerate(frozen):
        if value in seen:
            raise LabelValidationError(
                f"duplicate or equality-colliding label at positions {seen[value]} and {index}",
                path=name,
                context={"label": value},
            )
        seen[value] = index
    return frozen


P = TypeVar("P", bound="Profile")


@dataclass(frozen=True)
class Profile(Mapping[AgentId, TypeValue]):
    """An immutable agent-indexed type or report profile."""

    _items: tuple[tuple[AgentId, TypeValue], ...]

    @classmethod
    def from_mapping(
        cls: type[P],
        values: Mapping[Any, Any],
        *,
        agent_order: Iterable[Any] | None = None,
    ) -> P:
        """Build a profile, optionally enforcing a declared agent order."""

        if agent_order is None:
            raw_agents = tuple(values.keys())
        else:
            raw_agents = tuple(agent_order)
            expected = {freeze_label(agent) for agent in raw_agents}
            actual = {freeze_label(agent) for agent in values}
            if actual != expected:
                raise LabelValidationError(
                    "profile agents do not match the declared agent order",
                    context={"missing": expected - actual, "unexpected": actual - expected},
                )
        items = tuple(
            (
                freeze_label(agent, path="profile.agent"),
                freeze_label(values[agent], path=f"profile[{agent}]"),
            )
            for agent in raw_agents
        )
        return cls(items)

    @classmethod
    def from_values(cls: type[P], agent_order: Iterable[Any], values: Iterable[Any]) -> P:
        """Build a profile from parallel ordered agent and value axes."""

        agents = tuple(freeze_label(agent) for agent in agent_order)
        types = tuple(freeze_label(value) for value in values)
        if len(agents) != len(types):
            raise LabelValidationError("profile has the wrong number of values")
        return cls(tuple(zip(agents, types, strict=True)))

    def __post_init__(self) -> None:
        agents = [agent for agent, _ in self._items]
        if len(set(agents)) != len(agents):
            raise LabelValidationError("a profile contains duplicate agents")

    def __getitem__(self, agent: AgentId) -> TypeValue:
        for candidate, value in self._items:
            if candidate == agent:
                return value
        raise KeyError(agent)

    def __iter__(self) -> Iterator[AgentId]:
        return (agent for agent, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        body = ", ".join(f"{k!r}: {v!r}" for k, v in self._items)
        return f"{type(self).__name__}({{{body}}})"

    @property
    def values_tuple(self) -> tuple[TypeValue, ...]:
        """Values in the profile's canonical agent order."""

        return tuple(value for _, value in self._items)

    @property
    def agents_tuple(self) -> tuple[AgentId, ...]:
        """Agents in canonical order."""

        return tuple(agent for agent, _ in self._items)

    def with_report(self: P, agent: Any, report: Any) -> P:
        """Return a copy with one agent's type/report replaced."""

        frozen_agent = freeze_label(agent)
        frozen_report = freeze_label(report)
        if frozen_agent not in self:
            raise KeyError(agent)
        return type(self)(
            tuple(
                (candidate, frozen_report if candidate == frozen_agent else value)
                for candidate, value in self._items
            )
        )

    def without(self, agent: Any) -> tuple[tuple[AgentId, TypeValue], ...]:
        """Return the ordered profile of all agents except *agent*."""

        frozen_agent = freeze_label(agent)
        return tuple(item for item in self._items if item[0] != frozen_agent)

    def to_dict(self) -> dict[AgentId, TypeValue]:
        """Return a mutable mapping in canonical agent order."""

        return dict(self._items)


class TypeProfile(Profile):
    """A true type profile."""


class ReportProfile(Profile):
    """A direct-mechanism report profile."""


def coerce_profile(
    value: Any,
    *,
    agents: tuple[AgentId, ...],
    profile_type: type[P],
) -> P:
    """Coerce mapping, tuple, or one-agent shorthand into a profile."""

    if isinstance(value, Profile):
        if value.agents_tuple != agents:
            raise LabelValidationError("profile agent order does not match the domain")
        if isinstance(value, profile_type):
            return value
        return profile_type(value._items)
    if isinstance(value, Mapping):
        return profile_type.from_mapping(value, agent_order=agents)
    if len(agents) == 1 and not isinstance(value, tuple):
        return profile_type.from_values(agents, (value,))
    if isinstance(value, tuple):
        return profile_type.from_values(agents, value)
    raise LabelValidationError(
        "profile must be a mapping, an ordered tuple, or one-agent shorthand"
    )
