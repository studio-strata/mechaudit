"""Validated finite joint priors with labeled support."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import product
from numbers import Real
from types import MappingProxyType
from typing import Any

from mechaudit.core.labels import (
    AgentId,
    TypeProfile,
    TypeValue,
    coerce_profile,
    freeze_label,
    normalize_axis,
)
from mechaudit.exceptions import LabelValidationError, PriorValidationError


class MissingProfilePolicy(str, Enum):
    """How a prior constructor handles omitted Cartesian-product profiles."""

    ERROR = "error"
    ZERO = "zero"


def _probability(value: Any, *, path: str) -> float:
    if not isinstance(value, Real):
        raise PriorValidationError("probability must be a real number", path=path)
    result = float(value)
    if not math.isfinite(result):
        raise PriorValidationError("probability must be finite", path=path)
    if result < 0.0:
        raise PriorValidationError("probability cannot be negative", path=path)
    return result


def _marginal_tables(
    type_spaces: tuple[tuple[AgentId, tuple[TypeValue, ...]], ...],
    entries: tuple[tuple[TypeProfile, float], ...],
) -> Mapping[AgentId, Mapping[TypeValue, float]]:
    masses: dict[AgentId, dict[TypeValue, list[float]]] = {
        agent: {type_value: [] for type_value in types} for agent, types in type_spaces
    }
    for profile, mass in entries:
        for agent, type_value in profile._items:
            masses[agent][type_value].append(mass)
    return MappingProxyType(
        {
            agent: MappingProxyType(
                {type_value: sum(values) for type_value, values in by_type.items()}
            )
            for agent, by_type in masses.items()
        }
    )


def _missing_policy(value: Any) -> MissingProfilePolicy:
    try:
        return MissingProfilePolicy(value)
    except ValueError as exc:
        raise PriorValidationError(
            "missing must be 'error' or 'zero'", context={"missing": value}
        ) from exc


@dataclass(frozen=True, init=False, eq=False)
class JointPrior(Mapping[TypeProfile, float]):
    """An immutable probability distribution over a finite type product.

    Use the named constructors rather than relying on tuple axis positions.
    The stored probabilities are normalized only by the tiny floating-point
    residual accepted by ``normalization_tolerance``.
    """

    agents: tuple[AgentId, ...]
    type_spaces: tuple[tuple[AgentId, tuple[TypeValue, ...]], ...]
    _entries: tuple[tuple[TypeProfile, float], ...]
    normalization_error: float
    _index: Mapping[TypeProfile, float]
    _marginals: Mapping[AgentId, Mapping[TypeValue, float]]

    def __init__(
        self,
        *,
        agents: Sequence[Any],
        type_spaces: Mapping[Any, Iterable[Any]],
        probabilities: Mapping[Any, float],
        missing: Any = MissingProfilePolicy.ERROR,
        normalization_tolerance: float = 1e-12,
    ) -> None:
        built = type(self).from_mapping(
            probabilities,
            agent_order=agents,
            type_spaces=type_spaces,
            missing=missing,
            normalization_tolerance=normalization_tolerance,
        )
        object.__setattr__(self, "agents", built.agents)
        object.__setattr__(self, "type_spaces", built.type_spaces)
        object.__setattr__(self, "_entries", built._entries)
        object.__setattr__(self, "normalization_error", built.normalization_error)
        object.__setattr__(self, "_index", built._index)
        object.__setattr__(self, "_marginals", built._marginals)

    @classmethod
    def _create(
        cls,
        agents: tuple[AgentId, ...],
        type_spaces: tuple[tuple[AgentId, tuple[TypeValue, ...]], ...],
        entries: tuple[tuple[TypeProfile, float], ...],
        normalization_error: float,
    ) -> JointPrior:
        instance = object.__new__(cls)
        object.__setattr__(instance, "agents", agents)
        object.__setattr__(instance, "type_spaces", type_spaces)
        object.__setattr__(instance, "_entries", entries)
        object.__setattr__(instance, "normalization_error", normalization_error)
        object.__setattr__(instance, "_index", MappingProxyType(dict(entries)))
        object.__setattr__(instance, "_marginals", _marginal_tables(type_spaces, entries))
        return instance

    @classmethod
    def from_independent(
        cls,
        marginals: Mapping[Any, Mapping[Any, float]],
        *,
        normalization_tolerance: float = 1e-12,
    ) -> JointPrior:
        """Construct a joint prior as the product of labeled marginals."""

        if not marginals:
            raise PriorValidationError("at least one marginal is required")
        agents = normalize_axis(marginals.keys(), name="prior.agents")
        spaces: dict[AgentId, tuple[TypeValue, ...]] = {}
        normalized_marginals: dict[AgentId, dict[TypeValue, float]] = {}
        for raw_agent, agent in zip(marginals.keys(), agents, strict=True):
            marginal = marginals[raw_agent]
            if not isinstance(marginal, Mapping) or not marginal:
                raise PriorValidationError(
                    "each marginal must be a nonempty mapping",
                    context={"agent": agent},
                )
            types = normalize_axis(marginal.keys(), name=f"prior.types[{agent!r}]")
            values = {
                type_value: _probability(
                    probability,
                    path=f"prior.marginal[{agent!r}, {type_value!r}]",
                )
                for type_value, probability in zip(types, marginal.values(), strict=True)
            }
            total = sum(values.values())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=normalization_tolerance):
                raise PriorValidationError(
                    "marginal probabilities must sum to one",
                    context={"agent": agent, "sum": total},
                )
            values = {key: value / total for key, value in values.items()}
            spaces[agent] = types
            normalized_marginals[agent] = values

        probabilities: dict[TypeProfile, float] = {}
        for type_values in product(*(spaces[agent] for agent in agents)):
            profile = TypeProfile.from_values(agents, type_values)
            mass = math.prod(
                normalized_marginals[agent][type_value]
                for agent, type_value in zip(agents, type_values, strict=True)
            )
            probabilities[profile] = mass
        return cls._from_normalized_mapping(
            agents,
            spaces,
            probabilities,
            missing=MissingProfilePolicy.ERROR,
            normalization_tolerance=normalization_tolerance,
        )

    @classmethod
    def from_mapping(
        cls,
        probabilities: Mapping[Any, float],
        *,
        agent_order: Iterable[Any] | None = None,
        type_spaces: Mapping[Any, Iterable[Any]] | None = None,
        missing: Any = MissingProfilePolicy.ERROR,
        normalization_tolerance: float = 1e-12,
    ) -> JointPrior:
        """Construct a prior from profile-to-probability entries.

        ``agent_order`` is required unless keys are :class:`TypeProfile`
        instances. For a one-agent prior, scalar keys are accepted as a
        readable shorthand. Omitted Cartesian-product profiles are errors by
        default; pass ``missing="zero"`` to opt in to zero filling.
        """

        if not probabilities:
            raise PriorValidationError("a joint prior cannot be empty")
        if not math.isfinite(normalization_tolerance) or normalization_tolerance < 0:
            raise PriorValidationError("normalization_tolerance must be finite and nonnegative")

        if agent_order is None:
            first_key = next(iter(probabilities))
            if not isinstance(first_key, TypeProfile):
                raise PriorValidationError("agent_order is required for non-TypeProfile prior keys")
            agents = first_key.agents_tuple
        else:
            agents = normalize_axis(agent_order, name="prior.agents")

        parsed: dict[TypeProfile, float] = {}
        inferred: dict[AgentId, list[TypeValue]] = {agent: [] for agent in agents}
        for index, (raw_profile, raw_mass) in enumerate(probabilities.items()):
            try:
                profile = coerce_profile(raw_profile, agents=agents, profile_type=TypeProfile)
            except Exception as exc:
                if isinstance(exc, PriorValidationError):
                    raise
                raise PriorValidationError(
                    "invalid prior profile key", path=f"prior[{index}]"
                ) from exc
            if profile in parsed:
                raise PriorValidationError(
                    "duplicate prior profile after label normalization",
                    context={"profile": profile},
                )
            parsed[profile] = _probability(raw_mass, path=f"prior[{profile!r}]")
            for agent in agents:
                value = profile[agent]
                if value not in inferred[agent]:
                    inferred[agent].append(value)

        if type_spaces is None:
            spaces = {
                agent: normalize_axis(inferred[agent], name=f"prior.types[{agent!r}]")
                for agent in agents
            }
        else:
            frozen_space_keys = {freeze_label(key): key for key in type_spaces}
            if set(frozen_space_keys) != set(agents):
                raise PriorValidationError(
                    "type_spaces agents must match agent_order",
                    context={
                        "missing": set(agents) - set(frozen_space_keys),
                        "unexpected": set(frozen_space_keys) - set(agents),
                    },
                )
            spaces = {
                agent: normalize_axis(
                    type_spaces[frozen_space_keys[agent]],
                    name=f"prior.types[{agent!r}]",
                )
                for agent in agents
            }
        return cls._from_normalized_mapping(
            agents,
            spaces,
            parsed,
            missing=_missing_policy(missing),
            normalization_tolerance=normalization_tolerance,
        )

    @classmethod
    def _from_normalized_mapping(
        cls,
        agents: tuple[AgentId, ...],
        spaces: Mapping[AgentId, tuple[TypeValue, ...]],
        probabilities: Mapping[TypeProfile, float],
        *,
        missing: MissingProfilePolicy,
        normalization_tolerance: float,
    ) -> JointPrior:
        expected_profiles = tuple(
            TypeProfile.from_values(agents, values)
            for values in product(*(spaces[agent] for agent in agents))
        )
        expected_set = set(expected_profiles)
        unexpected = set(probabilities) - expected_set
        if unexpected:
            raise PriorValidationError(
                "prior contains profiles outside the declared type spaces",
                context={"unexpected": unexpected},
            )
        omitted = expected_set - set(probabilities)
        if omitted and missing is MissingProfilePolicy.ERROR:
            raise PriorValidationError(
                "prior omits Cartesian-product profiles; use missing='zero' explicitly",
                context={"missing_count": len(omitted), "examples": tuple(omitted)[:3]},
            )
        completed = {
            profile: float(probabilities.get(profile, 0.0)) for profile in expected_profiles
        }
        total = sum(completed.values())
        error = total - 1.0
        if total <= 0.0 or not math.isclose(
            total, 1.0, rel_tol=0.0, abs_tol=normalization_tolerance
        ):
            raise PriorValidationError(
                "joint probabilities must sum to one", context={"sum": total}
            )
        completed = {profile: mass / total for profile, mass in completed.items()}
        entries = tuple((profile, completed[profile]) for profile in expected_profiles)
        frozen_spaces = tuple((agent, spaces[agent]) for agent in agents)
        return cls._create(agents, frozen_spaces, entries, error)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        agent_order: Iterable[Any],
        probability_field: str = "probability",
        type_spaces: Mapping[Any, Iterable[Any]] | None = None,
        missing: Any = MissingProfilePolicy.ERROR,
        normalization_tolerance: float = 1e-12,
    ) -> JointPrior:
        """Construct a joint prior from records with agent-named columns."""

        agents = tuple(agent_order)
        probabilities: dict[TypeProfile, float] = {}
        for index, record in enumerate(records):
            if probability_field not in record:
                raise PriorValidationError(
                    "prior record is missing the probability field",
                    path=f"records[{index}]",
                )
            try:
                profile_values = {agent: record[agent] for agent in agents}
            except KeyError as exc:
                raise PriorValidationError(
                    "prior record is missing an agent column",
                    path=f"records[{index}]",
                ) from exc
            profile = TypeProfile.from_mapping(profile_values, agent_order=agents)
            if profile in probabilities:
                raise PriorValidationError(
                    "duplicate profile in prior records", context={"profile": profile}
                )
            probabilities[profile] = _probability(
                record[probability_field], path=f"records[{index}].{probability_field}"
            )
        return cls.from_mapping(
            probabilities,
            agent_order=agents,
            type_spaces=type_spaces,
            missing=missing,
            normalization_tolerance=normalization_tolerance,
        )

    @classmethod
    def from_xarray(
        cls,
        data: Any,
        *,
        normalization_tolerance: float = 1e-12,
    ) -> JointPrior:
        """Construct a prior from a labeled ``xarray.DataArray``.

        Xarray remains an optional dependency; the method uses its public
        ``dims``, ``coords``, and ``values`` interface without importing it.
        """

        if not all(hasattr(data, attribute) for attribute in ("dims", "coords", "values")):
            raise PriorValidationError("data is not an xarray-like labeled array")
        agents = tuple(data.dims)
        spaces = {agent: tuple(data.coords[agent].values.tolist()) for agent in agents}
        probabilities: dict[TypeProfile, float] = {}
        for indices in product(*(range(len(spaces[agent])) for agent in agents)):
            values = tuple(
                spaces[agent][index] for agent, index in zip(agents, indices, strict=True)
            )
            profile = TypeProfile.from_values(agents, values)
            probabilities[profile] = _probability(
                data.values[indices].item(), path=f"xarray{indices!r}"
            )
        return cls.from_mapping(
            probabilities,
            agent_order=agents,
            type_spaces=spaces,
            normalization_tolerance=normalization_tolerance,
        )

    def align(
        self,
        *,
        agents: Iterable[Any],
        type_spaces: Mapping[Any, Iterable[Any]],
    ) -> JointPrior:
        """Validate and reorder this prior against an environment domain."""

        expected_agents = normalize_axis(agents, name="agents")
        frozen_space_keys = {freeze_label(key): key for key in type_spaces}
        missing_spaces = [agent for agent in expected_agents if agent not in frozen_space_keys]
        if missing_spaces:
            raise PriorValidationError(
                "type_spaces must define every agent", context={"missing": missing_spaces}
            )
        expected_spaces = {
            agent: normalize_axis(type_spaces[frozen_space_keys[agent]], name=f"types[{agent!r}]")
            for agent in expected_agents
        }
        if set(self.agents) != set(expected_agents):
            raise PriorValidationError("prior agents do not match the environment")
        current_spaces = dict(self.type_spaces)
        for agent in expected_agents:
            if set(current_spaces[agent]) != set(expected_spaces[agent]):
                raise PriorValidationError(
                    "prior type space does not match the environment",
                    context={"agent": agent},
                )
        remapped: dict[TypeProfile, float] = {}
        for profile, mass in self._entries:
            remapped[
                TypeProfile.from_values(
                    expected_agents, (profile[agent] for agent in expected_agents)
                )
            ] = mass
        return type(self).from_mapping(
            remapped,
            agent_order=expected_agents,
            type_spaces=expected_spaces,
            missing=MissingProfilePolicy.ERROR,
        )

    def __getitem__(self, profile: Any) -> float:
        """Return the mass of a profile given as a profile, mapping, or tuple.

        One-agent priors also accept the bare type label, so ``prior["low"]``
        works as it does in the constructors.
        """

        if isinstance(profile, TypeProfile):
            return self._index[profile]
        if len(self.agents) == 1:
            try:
                shorthand = TypeProfile.from_values(self.agents, (freeze_label(profile),))
            except LabelValidationError:
                shorthand = None
            if shorthand is not None and shorthand in self._index:
                return self._index[shorthand]
        try:
            candidate = coerce_profile(profile, agents=self.agents, profile_type=TypeProfile)
        except (LabelValidationError, TypeError):
            raise KeyError(profile) from None
        try:
            return self._index[candidate]
        except KeyError:
            raise KeyError(profile) from None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JointPrior):
            return NotImplemented
        return (
            self.agents == other.agents
            and self.type_spaces == other.type_spaces
            and self._entries == other._entries
        )

    def __hash__(self) -> int:
        return hash((self.agents, self.type_spaces, self._entries))

    def __iter__(self) -> Iterator[TypeProfile]:
        return (profile for profile, _ in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def probabilities(self) -> Mapping[TypeProfile, float]:
        """Read-only profile-to-probability mapping."""

        return self._index

    def marginal(self, agent: Any, type_value: Any) -> float:
        """Return an agent's marginal probability for one type."""

        frozen_agent = freeze_label(agent)
        frozen_type = freeze_label(type_value)
        if frozen_agent not in self.agents:
            raise KeyError(agent)
        return self._marginals[frozen_agent].get(frozen_type, 0.0)

    def is_independent(self, *, tolerance: float = 1e-12) -> bool:
        """Whether every joint mass equals the product of its marginals.

        ``tolerance`` is an absolute bound on ``|p(t) - prod_i p_i(t_i)|``.
        One-agent priors are trivially independent.
        """

        if tolerance < 0 or not math.isfinite(tolerance):
            raise PriorValidationError("tolerance must be finite and nonnegative")
        if len(self.agents) == 1:
            return True
        for profile, mass in self._entries:
            product_mass = math.prod(
                self._marginals[agent][type_value] for agent, type_value in profile._items
            )
            if abs(mass - product_mass) > tolerance:
                return False
        return True

    def zero_marginal_types(
        self, *, tolerance: float = 0.0
    ) -> tuple[tuple[AgentId, TypeValue], ...]:
        """Return all agent/type pairs whose marginal is at most *tolerance*."""

        if tolerance < 0 or not math.isfinite(tolerance):
            raise PriorValidationError("tolerance must be finite and nonnegative")
        return tuple(
            (agent, type_value)
            for agent, types in self.type_spaces
            for type_value in types
            if self.marginal(agent, type_value) <= tolerance
        )
