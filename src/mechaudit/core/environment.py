"""Canonical finite mechanism-design environments."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from mechaudit.core.labels import (
    AgentId,
    OutcomeValue,
    ReportProfile,
    StructuredLabel,
    TypeProfile,
    TypeValue,
    freeze_label,
    normalize_axis,
)
from mechaudit.core.labels import (
    coerce_profile as _coerce_profile,
)
from mechaudit.core.prior import JointPrior
from mechaudit.core.valuations import InterdependentValues, ValuationKind, ValuationModel
from mechaudit.exceptions import DomainMismatchError, EnvironmentValidationError

if TYPE_CHECKING:
    from mechaudit.audit.report import MechanismAuditReport
    from mechaudit.design.problem import DesignProblem
    from mechaudit.mechanisms.direct import DirectMechanism
    from mechaudit.objectives import LinearObjective
    from mechaudit.options import AuditOptions
    from mechaudit.requirements.base import Requirement


class OtherProfileEntry(NamedTuple):
    """One profile of the other agents plus its position offset.

    ``base`` is the canonical profile index with the omitted agent's type
    position set to zero; adding ``type_position * stride`` completes it.
    """

    labels: tuple[tuple[AgentId, TypeValue], ...]
    base: int


def _fingerprint_value(value: Any) -> Any:
    """Return a type-tagged, JSON-compatible fingerprint representation."""

    if isinstance(value, StructuredLabel):
        return {"kind": "mapping", "value": value.to_dict()}
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "value": [
                [_fingerprint_value(key), _fingerprint_value(item)] for key, item in value.items()
            ],
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "value": [_fingerprint_value(item) for item in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": type(value).__name__, "value": value}
    return {
        "kind": "python",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


@dataclass(frozen=True, init=False, eq=False)
class FiniteEnvironment:
    """A validated, immutable finite direct-mechanism environment.

    Parameters
    ----------
    agents:
        Ordered mapping from agent labels to ordered finite type collections.
    outcomes:
        Ordered explicit social outcomes.
    prior:
        Optional joint prior. Belief-free DSIC and ex-post checks may omit it;
        Bayesian requirements and expected objectives require it.
    values:
        A :class:`~mechaudit.core.valuations.PrivateValues` or
        :class:`~mechaudit.core.valuations.InterdependentValues` provider.

    Notes
    -----
    Values are eagerly tabulated. This both validates callback coverage and
    prevents later callback mutation from changing a compiled environment.
    """

    agents: tuple[AgentId, ...]
    type_spaces: tuple[tuple[AgentId, tuple[TypeValue, ...]], ...]
    outcomes: tuple[OutcomeValue, ...]
    prior: JointPrior | None
    valuation_model: ValuationModel
    profiles: tuple[TypeProfile, ...]
    report_profiles: tuple[ReportProfile, ...]
    values: tuple[tuple[tuple[float, ...], ...], ...]
    domain_fingerprint: str
    _types: Mapping[AgentId, tuple[TypeValue, ...]]
    _agent_indices: Mapping[AgentId, int]
    _type_indices: Mapping[AgentId, Mapping[TypeValue, int]]
    _outcome_indices: Mapping[OutcomeValue, int]
    _profile_indices: Mapping[TypeProfile, int]
    _strides: tuple[int, ...]
    _type_counts: tuple[int, ...]
    _other_profiles: Mapping[AgentId, tuple[OtherProfileEntry, ...]]
    _profile_masses: tuple[float, ...] | None
    _value_array: NDArray[np.float64]

    def __init__(
        self,
        *,
        agents: Mapping[Any, Iterable[Any]],
        outcomes: Iterable[Any],
        prior: JointPrior | Mapping[Any, float] | None = None,
        values: ValuationModel | Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(agents, Mapping) or not agents:
            raise EnvironmentValidationError(
                "agents must be a nonempty ordered mapping of type spaces"
            )
        agent_axis = normalize_axis(agents.keys(), name="agents")
        raw_agent_keys = {freeze_label(key): key for key in agents}
        type_map = {
            agent: normalize_axis(agents[raw_agent_keys[agent]], name=f"types[{agent!r}]")
            for agent in agent_axis
        }
        outcome_axis = normalize_axis(outcomes, name="outcomes")

        value_provider = values
        if value_provider is None:
            raise EnvironmentValidationError("a valuation provider is required")
        if callable(value_provider) and not isinstance(value_provider, ValuationModel):
            value_provider = InterdependentValues(value_provider)
        if not isinstance(value_provider, ValuationModel):
            raise EnvironmentValidationError(
                "values must be wrapped in PrivateValues or InterdependentValues"
            )
        try:
            ValuationKind(value_provider.kind)
        except (AttributeError, ValueError) as exc:
            raise EnvironmentValidationError(
                "valuation model has an invalid kind declaration"
            ) from exc

        profiles = tuple(
            TypeProfile.from_values(agent_axis, type_values)
            for type_values in product(*(type_map[agent] for agent in agent_axis))
        )
        report_profiles = tuple(
            ReportProfile.from_values(agent_axis, profile.values_tuple) for profile in profiles
        )

        value_rows = []
        for agent in agent_axis:
            profile_rows = []
            for profile in profiles:
                outcome_values = []
                for outcome in outcome_axis:
                    try:
                        result = value_provider.value(agent, profile, outcome)
                    except EnvironmentValidationError:
                        raise
                    except Exception as exc:
                        raise EnvironmentValidationError(
                            "valuation callback failed",
                            path=f"values[{agent!r}, {profile!r}, {outcome!r}]",
                        ) from exc
                    numeric_result = float(result)
                    if not math.isfinite(numeric_result):
                        raise EnvironmentValidationError(
                            "valuation must be finite",
                            path=f"values[{agent!r}, {profile!r}, {outcome!r}]",
                        )
                    outcome_values.append(numeric_result)
                profile_rows.append(tuple(outcome_values))
            value_rows.append(tuple(profile_rows))

        aligned_prior = None
        if prior is not None:
            if isinstance(prior, JointPrior):
                aligned_prior = prior.align(agents=agent_axis, type_spaces=type_map)
            elif isinstance(prior, Mapping):
                aligned_prior = JointPrior.from_mapping(
                    prior,
                    agent_order=agent_axis,
                    type_spaces=type_map,
                    missing="error",
                )
            else:
                raise EnvironmentValidationError(
                    "prior must be a JointPrior, a labeled mapping, or None"
                )

        fingerprint_payload = {
            "agents": [_fingerprint_value(agent) for agent in agent_axis],
            "types": [
                {
                    "agent": _fingerprint_value(agent),
                    "values": [_fingerprint_value(value) for value in type_map[agent]],
                }
                for agent in agent_axis
            ],
            "outcomes": [_fingerprint_value(outcome) for outcome in outcome_axis],
        }
        payload = json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")

        type_indices = {
            agent: MappingProxyType(
                {type_value: index for index, type_value in enumerate(type_map[agent])}
            )
            for agent in agent_axis
        }
        object.__setattr__(self, "agents", agent_axis)
        object.__setattr__(
            self, "type_spaces", tuple((agent, type_map[agent]) for agent in agent_axis)
        )
        object.__setattr__(self, "outcomes", outcome_axis)
        object.__setattr__(self, "prior", aligned_prior)
        object.__setattr__(self, "valuation_model", value_provider)
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "report_profiles", report_profiles)
        object.__setattr__(self, "values", tuple(value_rows))
        object.__setattr__(self, "domain_fingerprint", hashlib.sha256(payload).hexdigest())
        object.__setattr__(self, "_types", MappingProxyType(type_map))
        object.__setattr__(
            self,
            "_agent_indices",
            MappingProxyType({agent: index for index, agent in enumerate(agent_axis)}),
        )
        object.__setattr__(self, "_type_indices", MappingProxyType(type_indices))
        object.__setattr__(
            self,
            "_outcome_indices",
            MappingProxyType({outcome: index for index, outcome in enumerate(outcome_axis)}),
        )
        object.__setattr__(
            self,
            "_profile_indices",
            MappingProxyType({profile: index for index, profile in enumerate(profiles)}),
        )

        # Profiles enumerate the Cartesian product with the last agent varying
        # fastest, so a profile's position is a mixed-radix number. Compilers
        # and audits use these tables instead of rebuilding labeled profiles.
        type_counts = tuple(len(type_map[agent]) for agent in agent_axis)
        strides = [1] * len(agent_axis)
        for position in range(len(agent_axis) - 2, -1, -1):
            strides[position] = strides[position + 1] * type_counts[position + 1]
        other_profiles: dict[AgentId, tuple[OtherProfileEntry, ...]] = {}
        for position, agent in enumerate(agent_axis):
            other_positions = tuple(index for index in range(len(agent_axis)) if index != position)
            entries = []
            for combination in product(*(range(type_counts[index]) for index in other_positions)):
                labels = tuple(
                    (agent_axis[index], type_map[agent_axis[index]][type_position])
                    for index, type_position in zip(other_positions, combination, strict=True)
                )
                base = sum(
                    type_position * strides[index]
                    for index, type_position in zip(other_positions, combination, strict=True)
                )
                entries.append(OtherProfileEntry(labels, base))
            other_profiles[agent] = tuple(entries)
        value_array = np.array(value_rows, dtype=np.float64).reshape(
            len(agent_axis), len(profiles), len(outcome_axis)
        )
        value_array.flags.writeable = False
        object.__setattr__(self, "_strides", tuple(strides))
        object.__setattr__(self, "_type_counts", type_counts)
        object.__setattr__(self, "_other_profiles", MappingProxyType(other_profiles))
        object.__setattr__(
            self,
            "_profile_masses",
            None
            if aligned_prior is None
            else tuple(aligned_prior[profile] for profile in profiles),
        )
        object.__setattr__(self, "_value_array", value_array)

    @property
    def types(self) -> Mapping[AgentId, tuple[TypeValue, ...]]:
        """Read-only mapping from each agent to its canonical type tuple."""

        return self._types

    @property
    def n_agents(self) -> int:
        """Number of agents."""

        return len(self.agents)

    @property
    def n_profiles(self) -> int:
        """Number of type/report profiles in the Cartesian product."""

        return len(self.profiles)

    @property
    def n_outcomes(self) -> int:
        """Number of explicit outcomes."""

        return len(self.outcomes)

    @property
    def valuation_kind(self) -> ValuationKind:
        """Canonical private/interdependent valuation declaration."""

        return ValuationKind(self.valuation_model.kind)

    @property
    def is_private_values(self) -> bool:
        """Whether private-value DSIC semantics are valid."""

        return self.valuation_kind is ValuationKind.PRIVATE

    def _type_position(self, agent_index: int, profile_index: int) -> int:
        """Return an agent's type position within a canonical profile index."""

        return (profile_index // self._strides[agent_index]) % self._type_counts[agent_index]

    def coerce_profile(self, profile: Any) -> TypeProfile:
        """Normalize and validate a true type profile against this domain."""

        # Mapping-valued one-agent types are otherwise ambiguous with a profile
        # mapping, so first test the complete object as the sole type label.
        if self.n_agents == 1:
            candidate = freeze_label(profile)
            agent = self.agents[0]
            if candidate in self._type_indices[agent]:
                result = TypeProfile.from_values(self.agents, (candidate,))
            elif (
                isinstance(profile, tuple)
                and len(profile) == 1
                and freeze_label(profile[0]) in self._type_indices[agent]
            ):
                result = TypeProfile.from_values(self.agents, (freeze_label(profile[0]),))
            else:
                result = _coerce_profile(profile, agents=self.agents, profile_type=TypeProfile)
        else:
            result = _coerce_profile(profile, agents=self.agents, profile_type=TypeProfile)
        if result not in self._profile_indices:
            raise DomainMismatchError(
                "profile contains a type outside the environment",
                context={"profile": result},
            )
        return result

    def coerce_report_profile(self, report: Any) -> ReportProfile:
        """Normalize and validate a report profile against this domain."""

        true_profile = self.coerce_profile(report)
        return ReportProfile(true_profile._items)

    def profile_index(self, profile: Any) -> int:
        """Return the canonical Cartesian-product index of a profile."""

        if isinstance(profile, ReportProfile):
            candidate = TypeProfile(profile._items)
        elif isinstance(profile, TypeProfile):
            candidate = profile
        else:
            candidate = self.coerce_profile(profile)
        try:
            return self._profile_indices[candidate]
        except KeyError as exc:
            raise DomainMismatchError(
                "profile contains a type outside the environment",
                context={"profile": candidate},
            ) from exc

    def agent_index(self, agent: Any) -> int:
        """Return an agent's canonical axis index."""

        candidate = freeze_label(agent)
        try:
            return self._agent_indices[candidate]
        except KeyError as exc:
            raise DomainMismatchError("unknown agent", context={"agent": candidate}) from exc

    def type_index(self, agent: Any, type_value: Any) -> int:
        """Return a type's canonical index on one agent's axis."""

        candidate_agent = freeze_label(agent)
        candidate_type = freeze_label(type_value)
        try:
            return self._type_indices[candidate_agent][candidate_type]
        except KeyError as exc:
            raise DomainMismatchError(
                "unknown agent/type coordinate",
                context={"agent": candidate_agent, "type": candidate_type},
            ) from exc

    def outcome_index(self, outcome: Any) -> int:
        """Return an outcome's canonical axis index."""

        candidate = freeze_label(outcome)
        try:
            return self._outcome_indices[candidate]
        except KeyError as exc:
            raise DomainMismatchError("unknown outcome", context={"outcome": candidate}) from exc

    def evaluate_value(self, agent: Any, profile: Any, outcome: Any) -> float:
        """Read a tabulated value using labeled coordinates."""

        return self.values[self.agent_index(agent)][self.profile_index(profile)][
            self.outcome_index(outcome)
        ]

    def with_prior(self, prior: JointPrior | Mapping[Any, float] | None) -> FiniteEnvironment:
        """Return the same domain and valuation model with a different prior."""

        return type(self)(
            agents=self.types,
            outcomes=self.outcomes,
            prior=prior,
            values=self.valuation_model,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FiniteEnvironment):
            return NotImplemented
        return (
            self.agents == other.agents
            and self.type_spaces == other.type_spaces
            and self.outcomes == other.outcomes
            and self.prior == other.prior
            and self.valuation_kind == other.valuation_kind
            and self.values == other.values
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.agents,
                self.type_spaces,
                self.outcomes,
                self.prior,
                self.valuation_kind,
                self.values,
            )
        )

    def audit(
        self,
        mechanism: DirectMechanism,
        requirements: Iterable[Requirement],
        *,
        options: AuditOptions | None = None,
    ) -> MechanismAuditReport:
        """Audit a fixed mechanism via the lazily imported audit engine."""

        audit_module = importlib.import_module("mechaudit.audit.engine")
        return cast(
            "MechanismAuditReport",
            audit_module.audit_mechanism(
                environment=self,
                mechanism=mechanism,
                requirements=tuple(requirements),
                options=options,
            ),
        )

    def design(
        self,
        *,
        objective: LinearObjective,
        requirements: Iterable[Requirement] = (),
        randomized: bool = True,
    ) -> DesignProblem:
        """Create a design problem without importing the solver at startup."""

        problem_module = importlib.import_module("mechaudit.design.problem")
        return cast(
            "DesignProblem",
            problem_module.DesignProblem(
                environment=self,
                objective=objective,
                requirements=tuple(requirements),
                randomized=randomized,
            ),
        )
