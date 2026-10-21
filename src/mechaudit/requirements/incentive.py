"""Distinct incentive-compatibility requirement declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def _zero_policy(value: Any) -> ZeroProbabilityPolicy:
    try:
        return ZeroProbabilityPolicy(value)
    except ValueError as exc:
        raise RequirementValidationError(
            "zero_probability_types must be 'error', 'vacuous', or 'enforce_ex_post'"
        ) from exc


@dataclass(frozen=True, init=False)
class DSIC(Requirement):
    """Private-value dominant-strategy incentive compatibility.

    Other agents' reports range independently. Compilers must reject this
    requirement for an environment declared to have interdependent values.
    """

    agents: Selector
    true_types: SelectorMap
    deviations: SelectorMap

    def __init__(
        self,
        *,
        agents: SelectorLike = None,
        true_types: SelectorMapLike = None,
        deviations: SelectorMapLike = None,
    ) -> None:
        object.__setattr__(self, "agents", freeze_selector(agents))
        object.__setattr__(self, "true_types", freeze_selector_map(true_types))
        object.__setattr__(self, "deviations", freeze_selector_map(deviations))

    @property
    def family(self) -> str:
        return "dsic"


@dataclass(frozen=True, init=False)
class ExPostIC(Requirement):
    """Ex-post IC with truthful reports by the other agents.

    Unlike :class:`DSIC`, values may depend on the complete true type profile.
    """

    agents: Selector
    true_types: SelectorMap
    deviations: SelectorMap

    def __init__(
        self,
        *,
        agents: SelectorLike = None,
        true_types: SelectorMapLike = None,
        deviations: SelectorMapLike = None,
    ) -> None:
        object.__setattr__(self, "agents", freeze_selector(agents))
        object.__setattr__(self, "true_types", freeze_selector_map(true_types))
        object.__setattr__(self, "deviations", freeze_selector_map(deviations))

    @property
    def family(self) -> str:
        return "ex_post_ic"


@dataclass(frozen=True, init=False)
class BIC(Requirement):
    """Bayesian incentive compatibility under an arbitrary finite joint prior.

    The compiler should form joint-probability-weighted rows and retain the
    marginal mass needed to report conditional utility slack in economic units.
    """

    agents: Selector
    true_types: SelectorMap
    deviations: SelectorMap
    zero_probability_types: ZeroProbabilityPolicy

    def __init__(
        self,
        *,
        agents: SelectorLike = None,
        true_types: SelectorMapLike = None,
        deviations: SelectorMapLike = None,
        zero_probability_types: ZeroProbabilityPolicy | str = ZeroProbabilityPolicy.ERROR,
    ) -> None:
        object.__setattr__(self, "agents", freeze_selector(agents))
        object.__setattr__(self, "true_types", freeze_selector_map(true_types))
        object.__setattr__(self, "deviations", freeze_selector_map(deviations))
        object.__setattr__(
            self,
            "zero_probability_types",
            _zero_policy(zero_probability_types),
        )

    @property
    def family(self) -> str:
        return "bic"


PrivateValueDSIC = DSIC
