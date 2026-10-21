"""Independent utility-level evaluation of fixed finite mechanisms."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from mechaudit.audit.report import (
    CheckStatus,
    ConstraintCheck,
    MechanismAuditReport,
)
from mechaudit.core.environment import FiniteEnvironment
from mechaudit.core.labels import AgentId, ReportProfile, TypeValue
from mechaudit.diagnostics.correlation import correlation_diagnostics
from mechaudit.diagnostics.health import NumericalHealthReport
from mechaudit.exceptions import (
    EnvironmentValidationError,
    RequirementValidationError,
    UnsupportedRequirementError,
    ZeroProbabilityTypeError,
)
from mechaudit.options import AuditOptions
from mechaudit.requirements.base import (
    AllocationCoordinate,
    ConstraintRelation,
    ConstraintSpec,
    PaymentCoordinate,
    ZeroProbabilityPolicy,
    format_constraint_id,
)
from mechaudit.requirements.budget import (
    BudgetTiming,
    NoSubsidies,
    PaymentBounds,
    StrongBudgetBalance,
    WeakBudgetBalance,
)
from mechaudit.requirements.custom import (
    CustomLinearConstraint,
    CustomLinearRequirement,
)
from mechaudit.requirements.feasibility import Capacity, Symmetry
from mechaudit.requirements.incentive import BIC, DSIC, ExPostIC
from mechaudit.requirements.participation import ExAnteIR, ExPostIR, InterimIR


def _selected(values: tuple[Any, ...] | None, value: Any) -> bool:
    return values is None or value in values


def _selected_for_agent(
    values: tuple[tuple[AgentId, tuple[TypeValue, ...]], ...] | None,
    agent: AgentId,
    value: TypeValue,
) -> bool:
    if values is None:
        return True
    mapping = dict(values)
    return agent not in mapping or value in mapping[agent]


def _inequality(
    family: str,
    qualifiers: Mapping[str, Any],
    slack: float,
    *,
    tolerance: float,
    unit: str,
    metadata: Mapping[str, Any] | None = None,
) -> ConstraintCheck:
    status = CheckStatus.PASS if slack >= -tolerance else CheckStatus.FAIL
    return ConstraintCheck(
        constraint_id=format_constraint_id(family, qualifiers),
        family=family,
        relation=">=",
        status=status,
        tolerance=tolerance,
        slack=float(slack),
        unit=unit,
        metadata={**qualifiers, **dict(metadata or {})},
    )


def _equality(
    family: str,
    qualifiers: Mapping[str, Any],
    residual: float,
    *,
    tolerance: float,
    unit: str,
    metadata: Mapping[str, Any] | None = None,
) -> ConstraintCheck:
    status = CheckStatus.PASS if abs(residual) <= tolerance else CheckStatus.FAIL
    return ConstraintCheck(
        constraint_id=format_constraint_id(family, qualifiers),
        family=family,
        relation="==",
        status=status,
        tolerance=tolerance,
        residual=float(residual),
        unit=unit,
        metadata={**qualifiers, **dict(metadata or {})},
    )


def _vacuous(
    family: str,
    qualifiers: Mapping[str, Any],
    *,
    tolerance: float,
    reason: str,
) -> ConstraintCheck:
    return ConstraintCheck(
        constraint_id=format_constraint_id(family, qualifiers),
        family=family,
        relation=">=",
        status=CheckStatus.VACUOUS,
        tolerance=tolerance,
        metadata={**qualifiers, "reason": reason},
    )


def _utility(
    environment: FiniteEnvironment,
    mechanism: Any,
    agent_index: int,
    true_index: int,
    report_index: int,
) -> float:
    values = environment._value_array[agent_index, true_index]
    return float(
        np.dot(mechanism.allocation[report_index], values)
        - mechanism.payments[report_index, agent_index]
    )


def _deviation_check(
    family: str,
    qualifiers: Mapping[str, Any],
    environment: FiniteEnvironment,
    mechanism: Any,
    *,
    agent_index: int,
    true_index: int,
    deviation_index: int,
    tolerance: float,
    extra_metadata: Mapping[str, Any] | None = None,
) -> ConstraintCheck:
    truthful = _utility(environment, mechanism, agent_index, true_index, true_index)
    deviating = _utility(environment, mechanism, agent_index, true_index, deviation_index)
    return _inequality(
        family,
        qualifiers,
        truthful - deviating,
        tolerance=tolerance,
        unit="utility",
        metadata={
            **dict(extra_metadata or {}),
            "truthful_utility": truthful,
            "deviation_utility": deviating,
            "deviation_gain": deviating - truthful,
        },
    )


def _audit_dsic(
    environment: FiniteEnvironment,
    mechanism: Any,
    requirement: DSIC,
    tolerance: float,
) -> list[ConstraintCheck]:
    if not environment.is_private_values:
        raise RequirementValidationError(
            "DSIC requires PrivateValues; use ExPostIC for an interdependent-value environment"
        )
    checks: list[ConstraintCheck] = []
    for agent_index, agent in enumerate(environment.agents):
        if not _selected(requirement.agents, agent):
            continue
        stride = environment._strides[agent_index]
        types = environment.types[agent]
        for true_position, true_type in enumerate(types):
            if not _selected_for_agent(requirement.true_types, agent, true_type):
                continue
            for deviation_position, deviation in enumerate(types):
                if deviation == true_type or not _selected_for_agent(
                    requirement.deviations, agent, deviation
                ):
                    continue
                for others, base in environment._other_profiles[agent]:
                    checks.append(
                        _deviation_check(
                            "dsic",
                            {
                                "agent": agent,
                                "true_type": true_type,
                                "deviation": deviation,
                                "others_report": others,
                            },
                            environment,
                            mechanism,
                            agent_index=agent_index,
                            true_index=base + true_position * stride,
                            deviation_index=base + deviation_position * stride,
                            tolerance=tolerance,
                        )
                    )
    return checks


def _audit_ex_post_ic(
    environment: FiniteEnvironment,
    mechanism: Any,
    requirement: ExPostIC,
    tolerance: float,
) -> list[ConstraintCheck]:
    checks: list[ConstraintCheck] = []
    for agent_index, agent in enumerate(environment.agents):
        if not _selected(requirement.agents, agent):
            continue
        stride = environment._strides[agent_index]
        types = environment.types[agent]
        for true_index, true_profile in enumerate(environment.profiles):
            true_position = environment._type_position(agent_index, true_index)
            true_type = types[true_position]
            if not _selected_for_agent(requirement.true_types, agent, true_type):
                continue
            for deviation_position, deviation in enumerate(types):
                if deviation == true_type or not _selected_for_agent(
                    requirement.deviations, agent, deviation
                ):
                    continue
                checks.append(
                    _deviation_check(
                        "ex_post_ic",
                        {
                            "agent": agent,
                            "true_type": true_type,
                            "deviation": deviation,
                            "true_profile": true_profile,
                        },
                        environment,
                        mechanism,
                        agent_index=agent_index,
                        true_index=true_index,
                        deviation_index=true_index + (deviation_position - true_position) * stride,
                        tolerance=tolerance,
                    )
                )
    return checks


def _audit_bic(
    environment: FiniteEnvironment,
    mechanism: Any,
    requirement: BIC,
    tolerance: float,
) -> list[ConstraintCheck]:
    prior = environment.prior
    masses = environment._profile_masses
    if prior is None or masses is None:
        raise EnvironmentValidationError("BIC auditing requires a joint prior")
    checks: list[ConstraintCheck] = []
    for agent_index, agent in enumerate(environment.agents):
        if not _selected(requirement.agents, agent):
            continue
        stride = environment._strides[agent_index]
        types = environment.types[agent]
        others_entries = environment._other_profiles[agent]
        for true_position, true_type in enumerate(types):
            if not _selected_for_agent(requirement.true_types, agent, true_type):
                continue
            marginal = prior.marginal(agent, true_type)
            for deviation_position, deviation in enumerate(types):
                if deviation == true_type or not _selected_for_agent(
                    requirement.deviations, agent, deviation
                ):
                    continue
                base_qualifiers = {
                    "agent": agent,
                    "true_type": true_type,
                    "deviation": deviation,
                }
                if marginal == 0.0:
                    if requirement.zero_probability_types is ZeroProbabilityPolicy.ERROR:
                        raise ZeroProbabilityTypeError(
                            "BIC is undefined for a zero-marginal type; choose vacuous or enforce_ex_post",
                            context={"agent": agent, "true_type": true_type},
                        )
                    if requirement.zero_probability_types is ZeroProbabilityPolicy.VACUOUS:
                        checks.append(
                            _vacuous(
                                "bic",
                                base_qualifiers,
                                tolerance=tolerance,
                                reason="zero_probability_type",
                            )
                        )
                        continue
                    for others, base in others_entries:
                        checks.append(
                            _deviation_check(
                                "bic",
                                {**base_qualifiers, "others_type": others},
                                environment,
                                mechanism,
                                agent_index=agent_index,
                                true_index=base + true_position * stride,
                                deviation_index=base + deviation_position * stride,
                                tolerance=tolerance,
                                extra_metadata={
                                    "zero_probability_policy": "enforce_ex_post",
                                    "conditioning_mass": marginal,
                                },
                            )
                        )
                    continue

                truthful_joint = 0.0
                deviating_joint = 0.0
                for _, base in others_entries:
                    true_index = base + true_position * stride
                    probability = masses[true_index]
                    truthful_joint += probability * _utility(
                        environment, mechanism, agent_index, true_index, true_index
                    )
                    deviating_joint += probability * _utility(
                        environment,
                        mechanism,
                        agent_index,
                        true_index,
                        base + deviation_position * stride,
                    )
                joint_slack = truthful_joint - deviating_joint
                truthful = truthful_joint / marginal
                deviating = deviating_joint / marginal
                checks.append(
                    _inequality(
                        "bic",
                        base_qualifiers,
                        joint_slack / marginal,
                        tolerance=tolerance,
                        unit="conditional_utility",
                        metadata={
                            "truthful_utility": truthful,
                            "deviation_utility": deviating,
                            "deviation_gain": deviating - truthful,
                            "joint_weighted_slack": joint_slack,
                            "conditioning_mass": marginal,
                        },
                    )
                )
    return checks


def _audit_ex_post_ir(
    environment: FiniteEnvironment,
    mechanism: Any,
    requirement: ExPostIR,
    tolerance: float,
) -> list[ConstraintCheck]:
    checks: list[ConstraintCheck] = []
    for agent_index, agent in enumerate(environment.agents):
        if not _selected(requirement.agents, agent):
            continue
        for profile_index, profile in enumerate(environment.profiles):
            true_type = profile[agent]
            if not _selected_for_agent(requirement.types, agent, true_type):
                continue
            utility = _utility(environment, mechanism, agent_index, profile_index, profile_index)
            outside = requirement.outside_option.value(agent, profile)
            checks.append(
                _inequality(
                    "ex_post_ir",
                    {"agent": agent, "true_profile": profile},
                    utility - outside,
                    tolerance=tolerance,
                    unit="utility",
                    metadata={
                        "true_type": true_type,
                        "truthful_utility": utility,
                        "outside_option": outside,
                        "information_rent": utility - outside,
                    },
                )
            )
    return checks


def _audit_interim_ir(
    environment: FiniteEnvironment,
    mechanism: Any,
    requirement: InterimIR,
    tolerance: float,
) -> list[ConstraintCheck]:
    prior = environment.prior
    masses = environment._profile_masses
    if prior is None or masses is None:
        raise EnvironmentValidationError("interim IR auditing requires a joint prior")
    checks: list[ConstraintCheck] = []
    for agent_index, agent in enumerate(environment.agents):
        if not _selected(requirement.agents, agent):
            continue
        stride = environment._strides[agent_index]
        others_entries = environment._other_profiles[agent]
        for type_position, type_value in enumerate(environment.types[agent]):
            if not _selected_for_agent(requirement.types, agent, type_value):
                continue
            marginal = prior.marginal(agent, type_value)
            qualifiers = {"agent": agent, "true_type": type_value}
            if marginal == 0.0:
                if requirement.zero_probability_types is ZeroProbabilityPolicy.ERROR:
                    raise ZeroProbabilityTypeError(
                        "interim IR is undefined for a zero-marginal type; choose vacuous or enforce_ex_post",
                        context=qualifiers,
                    )
                if requirement.zero_probability_types is ZeroProbabilityPolicy.VACUOUS:
                    checks.append(
                        _vacuous(
                            "interim_ir",
                            qualifiers,
                            tolerance=tolerance,
                            reason="zero_probability_type",
                        )
                    )
                    continue
                for others, base in others_entries:
                    profile_index = base + type_position * stride
                    utility = _utility(
                        environment, mechanism, agent_index, profile_index, profile_index
                    )
                    outside = requirement.outside_option.value(
                        agent, environment.profiles[profile_index]
                    )
                    checks.append(
                        _inequality(
                            "interim_ir",
                            {**qualifiers, "others_type": others},
                            utility - outside,
                            tolerance=tolerance,
                            unit="utility",
                            metadata={
                                "zero_probability_policy": "enforce_ex_post",
                                "truthful_utility": utility,
                                "outside_option": outside,
                                "information_rent": utility - outside,
                                "conditioning_mass": marginal,
                            },
                        )
                    )
                continue

            utility_joint = 0.0
            outside_joint = 0.0
            for _, base in others_entries:
                profile_index = base + type_position * stride
                probability = masses[profile_index]
                utility_joint += probability * _utility(
                    environment, mechanism, agent_index, profile_index, profile_index
                )
                outside_joint += probability * requirement.outside_option.value(
                    agent, environment.profiles[profile_index]
                )
            utility = utility_joint / marginal
            outside = outside_joint / marginal
            checks.append(
                _inequality(
                    "interim_ir",
                    qualifiers,
                    utility - outside,
                    tolerance=tolerance,
                    unit="conditional_utility",
                    metadata={
                        "truthful_utility": utility,
                        "outside_option": outside,
                        "information_rent": utility - outside,
                        "joint_weighted_slack": utility_joint - outside_joint,
                        "conditioning_mass": marginal,
                    },
                )
            )
    return checks


def _audit_ex_ante_ir(
    environment: FiniteEnvironment,
    mechanism: Any,
    requirement: ExAnteIR,
    tolerance: float,
) -> list[ConstraintCheck]:
    masses = environment._profile_masses
    if environment.prior is None or masses is None:
        raise EnvironmentValidationError("ex-ante IR auditing requires a joint prior")
    checks: list[ConstraintCheck] = []
    for agent_index, agent in enumerate(environment.agents):
        if not _selected(requirement.agents, agent):
            continue
        utility = 0.0
        outside = 0.0
        for profile_index, profile in enumerate(environment.profiles):
            probability = masses[profile_index]
            utility += probability * _utility(
                environment, mechanism, agent_index, profile_index, profile_index
            )
            outside += probability * requirement.outside_option.value(agent, profile)
        checks.append(
            _inequality(
                "ex_ante_ir",
                {"agent": agent},
                utility - outside,
                tolerance=tolerance,
                unit="expected_utility",
                metadata={
                    "truthful_utility": utility,
                    "outside_option": outside,
                    "information_rent": utility - outside,
                },
            )
        )
    return checks


def _selected_reports(environment: FiniteEnvironment, reports: Any) -> tuple[ReportProfile, ...]:
    if reports is None:
        return environment.report_profiles
    return tuple(environment.coerce_report_profile(report) for report in reports)


def _audit_budget(
    environment: FiniteEnvironment,
    mechanism: Any,
    requirement: Any,
    tolerance: float,
) -> list[ConstraintCheck]:
    checks: list[ConstraintCheck] = []
    if isinstance(requirement, NoSubsidies):
        for report in _selected_reports(environment, requirement.reports):
            row = environment.profile_index(report)
            for agent in environment.agents:
                if not _selected(requirement.agents, agent):
                    continue
                payment = float(mechanism.payments[row, environment.agent_index(agent)])
                checks.append(
                    _inequality(
                        "no_subsidies",
                        {"agent": agent, "report": report},
                        payment,
                        tolerance=tolerance,
                        unit="money",
                    )
                )
        return checks
    if isinstance(requirement, PaymentBounds):
        for report in _selected_reports(environment, requirement.reports):
            row = environment.profile_index(report)
            for agent in environment.agents:
                if not _selected(requirement.agents, agent):
                    continue
                payment = float(mechanism.payments[row, environment.agent_index(agent)])
                if requirement.lower is not None:
                    checks.append(
                        _inequality(
                            "payment_bounds",
                            {"agent": agent, "report": report, "side": "lower"},
                            payment - requirement.lower,
                            tolerance=tolerance,
                            unit="money",
                            metadata={"bound": requirement.lower, "payment": payment},
                        )
                    )
                if requirement.upper is not None:
                    checks.append(
                        _inequality(
                            "payment_bounds",
                            {"agent": agent, "report": report, "side": "upper"},
                            requirement.upper - payment,
                            tolerance=tolerance,
                            unit="money",
                            metadata={"bound": requirement.upper, "payment": payment},
                        )
                    )
        return checks

    if isinstance(requirement, (WeakBudgetBalance, StrongBudgetBalance)):
        family = requirement.family
        target = (
            requirement.floor if isinstance(requirement, WeakBudgetBalance) else requirement.target
        )
        if requirement.timing is BudgetTiming.EX_POST:
            for report in environment.report_profiles:
                revenue = float(np.sum(mechanism.payments[environment.profile_index(report)]))
                if isinstance(requirement, WeakBudgetBalance):
                    checks.append(
                        _inequality(
                            family,
                            {"report": report},
                            revenue - target,
                            tolerance=tolerance,
                            unit="money",
                            metadata={"designer_revenue": revenue, "floor": target},
                        )
                    )
                else:
                    checks.append(
                        _equality(
                            family,
                            {"report": report},
                            revenue - target,
                            tolerance=tolerance,
                            unit="money",
                            metadata={"designer_revenue": revenue, "target": target},
                        )
                    )
        else:
            masses = environment._profile_masses
            if environment.prior is None or masses is None:
                raise EnvironmentValidationError("ex-ante budget balance requires a joint prior")
            revenue = math.fsum(
                probability * float(np.sum(mechanism.payments[report_index]))
                for report_index, probability in enumerate(masses)
            )
            if isinstance(requirement, WeakBudgetBalance):
                checks.append(
                    _inequality(
                        family,
                        {"timing": "ex_ante"},
                        revenue - target,
                        tolerance=tolerance,
                        unit="expected_money",
                        metadata={"designer_revenue": revenue, "floor": target},
                    )
                )
            else:
                checks.append(
                    _equality(
                        family,
                        {"timing": "ex_ante"},
                        revenue - target,
                        tolerance=tolerance,
                        unit="expected_money",
                        metadata={"designer_revenue": revenue, "target": target},
                    )
                )
        return checks
    raise UnsupportedRequirementError(f"unsupported budget requirement: {requirement!r}")


def _evaluate_constraint_spec(
    environment: FiniteEnvironment, mechanism: Any, spec: ConstraintSpec
) -> float:
    value = spec.expression.constant
    for coordinate, coefficient in spec.expression.terms:
        if isinstance(coordinate, AllocationCoordinate):
            row = environment.profile_index(coordinate.report)
            column = environment.outcome_index(coordinate.outcome)
            value += coefficient * float(mechanism.allocation[row, column])
        elif isinstance(coordinate, PaymentCoordinate):
            row = environment.profile_index(coordinate.report)
            column = environment.agent_index(coordinate.agent)
            value += coefficient * float(mechanism.payments[row, column])
        else:  # pragma: no cover - guarded by LinearExpression validation
            raise UnsupportedRequirementError("unknown custom mechanism coordinate")
    return float(value)


def _audit_custom(
    environment: FiniteEnvironment,
    mechanism: Any,
    requirement: CustomLinearConstraint | CustomLinearRequirement,
    tolerance: float,
) -> list[ConstraintCheck]:
    rows = (
        (requirement,)
        if isinstance(requirement, CustomLinearConstraint)
        else requirement.constraints
    )
    checks = []
    for row in rows:
        value = _evaluate_constraint_spec(environment, mechanism, row.spec)
        qualifiers = dict(row.spec.id.qualifiers)
        if row.spec.relation is ConstraintRelation.EQUAL:
            checks.append(
                _equality(
                    row.family,
                    qualifiers,
                    value,
                    tolerance=tolerance,
                    unit=row.spec.economic_unit,
                )
            )
        else:
            checks.append(
                _inequality(
                    row.family,
                    qualifiers,
                    value,
                    tolerance=tolerance,
                    unit=row.spec.economic_unit,
                )
            )
    return checks


def _audit_capacity(
    environment: FiniteEnvironment,
    mechanism: Any,
    requirement: Capacity,
    tolerance: float,
) -> list[ConstraintCheck]:
    usage = {environment.outcome_index(outcome): amount for outcome, amount in requirement.usage}
    reports = environment.report_profiles
    if requirement.profiles is not None:
        reports = tuple(environment.coerce_report_profile(item) for item in requirement.profiles)
    checks: list[ConstraintCheck] = []
    for report in reports:
        row = environment.profile_index(report)
        amount = math.fsum(
            coefficient * float(mechanism.allocation[row, outcome])
            for outcome, coefficient in usage.items()
        )
        checks.append(
            _inequality(
                "capacity",
                {"resource": requirement.resource, "report": report, "side": "upper"},
                requirement.upper - amount,
                tolerance=tolerance,
                unit="resource",
                metadata={"usage": amount, "bound": requirement.upper},
            )
        )
        if requirement.lower is not None:
            checks.append(
                _inequality(
                    "capacity",
                    {"resource": requirement.resource, "report": report, "side": "lower"},
                    amount - requirement.lower,
                    tolerance=tolerance,
                    unit="resource",
                    metadata={"usage": amount, "bound": requirement.lower},
                )
            )
    return checks


def _audit_symmetry(
    environment: FiniteEnvironment,
    mechanism: Any,
    requirement: Symmetry,
    tolerance: float,
) -> list[ConstraintCheck]:
    checks: list[ConstraintCheck] = []
    for operation in requirement.operations:
        agent_map = dict(operation.agents)
        type_maps = {agent: dict(mapping) for agent, mapping in operation.types}
        outcome_map = dict(operation.outcomes)
        if set(agent_map) != set(environment.agents) or set(agent_map.values()) != set(
            environment.agents
        ):
            raise RequirementValidationError(
                "symmetry agent action must be a permutation of all environment agents"
            )
        if set(outcome_map) != set(environment.outcomes) or set(outcome_map.values()) != set(
            environment.outcomes
        ):
            raise RequirementValidationError(
                "symmetry outcome action must be a permutation of all outcomes"
            )
        if set(type_maps) != set(environment.agents):
            raise RequirementValidationError(
                "symmetry type action must define every environment agent exactly once"
            )
        for source, target in agent_map.items():
            mapping = type_maps[source]
            if (
                set(mapping) != set(environment.types[source])
                or set(mapping.values()) != set(environment.types[target])
                or len(set(mapping.values())) != len(mapping)
            ):
                raise RequirementValidationError(
                    "symmetry type action must be a bijection between mapped type spaces",
                    context={
                        "operation": operation.name,
                        "source_agent": source,
                        "target_agent": target,
                    },
                )
        for report in environment.report_profiles:
            transformed_values: dict[AgentId, TypeValue] = {}
            for source in environment.agents:
                target = agent_map[source]
                report_mapping = type_maps.get(source)
                if report_mapping is None or report[source] not in report_mapping:
                    raise RequirementValidationError(
                        "symmetry type action is incomplete",
                        context={"operation": operation.name, "agent": source},
                    )
                transformed_values[target] = report_mapping[report[source]]
            transformed = environment.coerce_report_profile(transformed_values)
            original_row = environment.profile_index(report)
            transformed_row = environment.profile_index(transformed)
            for outcome in environment.outcomes:
                residual = float(
                    mechanism.allocation[original_row, environment.outcome_index(outcome)]
                    - mechanism.allocation[
                        transformed_row, environment.outcome_index(outcome_map[outcome])
                    ]
                )
                checks.append(
                    _equality(
                        "symmetry",
                        {
                            "operation": operation.name,
                            "report": report,
                            "outcome": outcome,
                            "component": "allocation",
                        },
                        residual,
                        tolerance=tolerance,
                        unit="probability",
                    )
                )
            for source in environment.agents:
                target = agent_map[source]
                residual = float(
                    mechanism.payments[original_row, environment.agent_index(source)]
                    - mechanism.payments[transformed_row, environment.agent_index(target)]
                )
                checks.append(
                    _equality(
                        "symmetry",
                        {
                            "operation": operation.name,
                            "report": report,
                            "agent": source,
                            "component": "payment",
                        },
                        residual,
                        tolerance=tolerance,
                        unit="money",
                    )
                )
    return checks


def _numerical_health(
    environment: FiniteEnvironment,
    mechanism: Any,
    checks: Iterable[ConstraintCheck],
    options: AuditOptions,
) -> NumericalHealthReport:
    checks_tuple = tuple(checks)
    allocation = np.asarray(mechanism.allocation, dtype=float)
    payments = np.asarray(mechanism.payments, dtype=float)
    values = np.asarray(environment.values, dtype=float)
    coefficients = np.concatenate(
        [np.abs(values).ravel(), np.abs(allocation).ravel(), np.abs(payments).ravel()]
    )
    positive = coefficients[coefficients > 0.0]
    allocation_residual = float(
        max(
            np.max(np.maximum(-allocation, 0.0), initial=0.0),
            np.max(np.maximum(allocation - 1.0, 0.0), initial=0.0),
        )
    )
    normalization_residual = float(np.max(np.abs(np.sum(allocation, axis=1) - 1.0)))
    ic = [check for check in checks_tuple if check.family in {"dsic", "ex_post_ic", "bic"}]
    ir = [
        check
        for check in checks_tuple
        if check.family in {"ex_post_ir", "interim_ir", "ex_ante_ir"}
    ]
    largest_ic = max((max(0.0, -(check.slack or 0.0)) for check in ic), default=0.0)
    largest_ir = max((max(0.0, -(check.slack or 0.0)) for check in ir), default=0.0)

    smallest_prior = None
    smallest_marginal = None
    warnings: list[str] = []
    details: dict[str, Any] = {}
    condition = None
    if environment.prior is not None:
        masses = [mass for mass in environment.prior.values() if mass > 0.0]
        smallest_prior = min(masses) if masses else None
        marginals = [
            marginal
            for agent in environment.agents
            for value in environment.types[agent]
            if (marginal := environment.prior.marginal(agent, value)) > 0.0
        ]
        smallest_marginal = min(marginals) if marginals else None
        # Under an independent prior every agent's conditional beliefs are
        # identical across own types. That rank deficiency is the definition of
        # independence, not a numerical concern, so belief conditioning is
        # diagnosed only for correlated priors.
        independent = environment.prior.is_independent(
            tolerance=max(options.tolerances.probability, 1e-12)
        )
        details["independent_prior"] = independent
        if not independent:
            try:
                belief_reports = correlation_diagnostics(environment)
                condition = max((item.condition_number for item in belief_reports), default=None)
                details["conditional_beliefs"] = [item.to_dict() for item in belief_reports]
                for item in belief_reports:
                    warnings.extend(item.warnings)
            except np.linalg.LinAlgError:
                warnings.append("Conditional-belief singular-value decomposition did not converge.")
    value_scale = float(np.max(np.abs(values), initial=0.0))
    payment_scale = float(np.max(np.abs(payments), initial=0.0))
    payment_ratio = (
        math.inf
        if value_scale == 0.0 and payment_scale > 0.0
        else (payment_scale / value_scale if value_scale > 0.0 else 0.0)
    )
    if allocation_residual > options.tolerances.feasibility:
        warnings.append("Allocation probabilities leave the [0, 1] range.")
    if normalization_residual > options.tolerances.feasibility:
        warnings.append("Allocation probabilities do not normalize at every report profile.")
    if smallest_prior is not None and smallest_prior <= options.tolerances.probability:
        warnings.append(
            "The joint prior contains positive masses at or below the probability tolerance."
        )
    if smallest_marginal is not None and smallest_marginal <= options.tolerances.probability:
        warnings.append("A type has positive marginal mass at or below the probability tolerance.")
    if payment_ratio > 1e6:
        warnings.append("Payments are more than one million times the environment's value scale.")
    return NumericalHealthReport(
        coefficient_min_positive=float(np.min(positive)) if positive.size else None,
        coefficient_max=float(np.max(coefficients)) if coefficients.size else None,
        smallest_positive_prior=smallest_prior,
        smallest_type_marginal=smallest_marginal,
        largest_allocation_residual=allocation_residual,
        largest_normalization_residual=normalization_residual,
        largest_ic_violation=largest_ic,
        largest_ir_violation=largest_ir,
        payment_to_value_ratio=payment_ratio,
        condition_estimate=condition,
        warnings=tuple(dict.fromkeys(warnings)),
        details=details,
    )


def audit_mechanism(
    *,
    environment: FiniteEnvironment,
    mechanism: Any,
    requirements: Iterable[Any] = (),
    options: AuditOptions | None = None,
) -> MechanismAuditReport:
    """Exhaustively audit a supplied mechanism in original economic units."""

    if not isinstance(environment, FiniteEnvironment):
        raise TypeError("environment must be a FiniteEnvironment")
    audit_options = options or AuditOptions()
    if not isinstance(audit_options, AuditOptions):
        raise TypeError("options must be an AuditOptions object")
    bound = mechanism.bind(environment)
    tolerance = audit_options.tolerances.feasibility
    probability_tolerance = audit_options.tolerances.probability
    checks: list[ConstraintCheck] = []

    # Baseline outcome-probability validity is always visible and is not a
    # user-selectable policy restriction.
    for report in environment.report_profiles:
        row = environment.profile_index(report)
        for outcome in environment.outcomes:
            probability = float(bound.allocation[row, environment.outcome_index(outcome)])
            checks.append(
                _inequality(
                    "allocation_nonnegative",
                    {"report": report, "outcome": outcome},
                    probability,
                    tolerance=tolerance,
                    unit="probability",
                    metadata={"probability": probability},
                )
            )
        checks.append(
            _equality(
                "allocation_normalization",
                {"report": report},
                float(np.sum(bound.allocation[row]) - 1.0),
                tolerance=tolerance,
                unit="probability",
            )
        )

    declared = tuple(requirements)
    for requirement in declared:
        if isinstance(requirement, DSIC):
            checks.extend(_audit_dsic(environment, bound, requirement, tolerance))
        elif isinstance(requirement, ExPostIC):
            checks.extend(_audit_ex_post_ic(environment, bound, requirement, tolerance))
        elif isinstance(requirement, BIC):
            checks.extend(_audit_bic(environment, bound, requirement, tolerance))
        elif isinstance(requirement, ExPostIR):
            checks.extend(_audit_ex_post_ir(environment, bound, requirement, tolerance))
        elif isinstance(requirement, InterimIR):
            checks.extend(_audit_interim_ir(environment, bound, requirement, tolerance))
        elif isinstance(requirement, ExAnteIR):
            checks.extend(_audit_ex_ante_ir(environment, bound, requirement, tolerance))
        elif isinstance(
            requirement,
            (NoSubsidies, PaymentBounds, WeakBudgetBalance, StrongBudgetBalance),
        ):
            checks.extend(_audit_budget(environment, bound, requirement, tolerance))
        elif isinstance(requirement, (CustomLinearConstraint, CustomLinearRequirement)):
            checks.extend(_audit_custom(environment, bound, requirement, tolerance))
        elif isinstance(requirement, Capacity):
            checks.extend(_audit_capacity(environment, bound, requirement, tolerance))
        elif isinstance(requirement, Symmetry):
            checks.extend(_audit_symmetry(environment, bound, requirement, tolerance))
        else:
            raise UnsupportedRequirementError(
                f"unsupported audit requirement: {type(requirement).__name__}"
            )

    ids = [check.constraint_id for check in checks]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise RequirementValidationError(
            "requirements generate duplicate semantic constraint ids",
            context={"duplicates": duplicates[:10]},
        )
    health = _numerical_health(environment, bound, checks, audit_options)
    return MechanismAuditReport(
        checks=tuple(checks),
        environment_fingerprint=environment.domain_fingerprint,
        mechanism_fingerprint=bound.fingerprint,
        feasibility_tolerance=tolerance,
        probability_tolerance=probability_tolerance,
        numerical_health=health,
        requirements=tuple(type(requirement).__name__ for requirement in declared),
    )
