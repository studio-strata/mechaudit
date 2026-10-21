"""Sparse compilation of finite randomized direct-mechanism design LPs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from mechaudit.core.environment import FiniteEnvironment
from mechaudit.core.labels import AgentId, ReportProfile, TypeValue
from mechaudit.design.indexing import VariableIndex, VariableIndexBuilder, VariableKey
from mechaudit.design.linear import LinearExpression, LinearExpressionBuilder
from mechaudit.design.sparse_model import (
    ColumnRecord,
    CompiledLP,
    RowRecord,
    RowScale,
    RowSense,
)
from mechaudit.exceptions import (
    EnvironmentValidationError,
    RequirementValidationError,
    UnsupportedRequirementError,
    ZeroProbabilityTypeError,
)
from mechaudit.mechanisms.direct import DirectMechanism
from mechaudit.objectives import (
    CustomLinearObjective,
    ExpectedResidualSurplus,
    ExpectedRevenue,
    ExpectedSocialWelfare,
    LinearCombination,
    LinearObjective,
    WeightedAgentUtilities,
)
from mechaudit.requirements.base import (
    AllocationCoordinate,
    ConstraintRelation,
    PaymentCoordinate,
    ZeroProbabilityPolicy,
    format_constraint_id,
    qualifier_text,
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


def _metadata(qualifiers: Mapping[str, Any]) -> dict[str, Any]:
    """Convert arbitrary economic labels to fingerprint-safe display metadata."""

    result: dict[str, Any] = {}
    for key, value in qualifiers.items():
        if value is None or isinstance(value, (bool, int, float, str)):
            result[key] = value
        else:
            result[key] = qualifier_text(value)
    return result


def _equilibration_scale(expression: LinearExpression, rhs: float) -> float:
    magnitudes = [abs(coefficient) for _, coefficient in expression.terms]
    magnitudes.append(abs(rhs))
    maximum = max(magnitudes, default=0.0)
    if maximum == 0.0:
        return 1.0
    exponent = round(-math.log2(maximum))
    exponent = max(-30, min(30, exponent))
    return math.ldexp(1.0, exponent)


def _make_scale(
    expression: LinearExpression,
    rhs: float,
    *,
    report_factor: float = 1.0,
    original_unit: str = "utility",
    report_unit: str | None = None,
) -> RowScale:
    return RowScale(
        factor=_equilibration_scale(expression, rhs),
        report_factor=report_factor,
        original_unit=original_unit,
        report_unit=report_unit or original_unit,
    )


@dataclass(frozen=True)
class MechanismVariableLayout:
    """Deterministic allocation/payment column layout."""

    index: VariableIndex
    n_profiles: int
    n_outcomes: int
    n_agents: int

    @classmethod
    def create(cls, environment: FiniteEnvironment) -> MechanismVariableLayout:
        builder = VariableIndexBuilder()
        for report_index in range(environment.n_profiles):
            for outcome_index in range(environment.n_outcomes):
                builder.add("allocation", report_index, outcome_index)
        for report_index in range(environment.n_profiles):
            for agent_index in range(environment.n_agents):
                builder.add("payment", report_index, agent_index)
        return cls(
            builder.freeze(),
            environment.n_profiles,
            environment.n_outcomes,
            environment.n_agents,
        )

    @property
    def n_columns(self) -> int:
        return len(self.index)

    def allocation(self, report_index: int, outcome_index: int) -> int:
        # ``create`` adds allocation columns row-major, then payment columns.
        if not (0 <= report_index < self.n_profiles and 0 <= outcome_index < self.n_outcomes):
            raise KeyError(VariableKey("allocation", (report_index, outcome_index)))
        return report_index * self.n_outcomes + outcome_index

    def payment(self, report_index: int, agent_index: int) -> int:
        if not (0 <= report_index < self.n_profiles and 0 <= agent_index < self.n_agents):
            raise KeyError(VariableKey("payment", (report_index, agent_index)))
        return self.n_profiles * self.n_outcomes + report_index * self.n_agents + agent_index

    @property
    def allocation_slice(self) -> slice:
        return slice(0, self.n_profiles * self.n_outcomes)

    @property
    def payment_slice(self) -> slice:
        start = self.n_profiles * self.n_outcomes
        return slice(start, start + self.n_profiles * self.n_agents)

    def vector_from_mechanism(
        self, environment: FiniteEnvironment, mechanism: DirectMechanism
    ) -> np.ndarray:
        bound = mechanism.bind(environment)
        return np.concatenate((bound.allocation.ravel(), bound.payments.ravel()))

    def mechanism_from_vector(
        self, environment: FiniteEnvironment, vector: np.ndarray
    ) -> DirectMechanism:
        values = np.asarray(vector, dtype=float)
        if values.shape != (self.n_columns,) or not np.all(np.isfinite(values)):
            raise ValueError("solution vector has the wrong shape or nonfinite values")
        allocation = values[self.allocation_slice].reshape(self.n_profiles, self.n_outcomes)
        payments = values[self.payment_slice].reshape(self.n_profiles, self.n_agents)
        return DirectMechanism.from_arrays(
            allocation,
            payments,
            environment=environment,
        )


@dataclass(frozen=True)
class VacuousConstraint:
    """A conditional economic row omitted from the LP as vacuous."""

    key: str
    family: str
    reason: str
    metadata: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class CompiledDesign:
    """A compiled LP plus the semantic objects needed to decode it."""

    environment: FiniteEnvironment
    objective: LinearObjective
    requirements: tuple[Any, ...]
    layout: MechanismVariableLayout
    model: CompiledLP
    vacuous_constraints: tuple[VacuousConstraint, ...] = ()

    @property
    def fingerprint(self) -> str:
        return self.model.fingerprint

    @property
    def economic_row_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, record in enumerate(self.model.rows)
            if record.classifiable and record.sense is RowSense.GREATER_EQUAL
        )

    def rows_for_family(self, family: str) -> tuple[int, ...]:
        return tuple(
            index for index, record in enumerate(self.model.rows) if record.family == family
        )

    def vector_from_mechanism(self, mechanism: DirectMechanism) -> np.ndarray:
        return self.layout.vector_from_mechanism(self.environment, mechanism)

    def mechanism_from_vector(self, vector: np.ndarray) -> DirectMechanism:
        return self.layout.mechanism_from_vector(self.environment, vector)

    def objective_value(self, mechanism: DirectMechanism) -> float:
        return self.model.objective_value(self.vector_from_mechanism(mechanism))

    def workload_estimate(self, *, level: str = "full") -> dict[str, int]:
        inequalities = len(self.economic_row_indices)
        return {
            "economic_inequalities": inequalities,
            "principal_solves": 2,
            "maximum_slack_fallbacks": inequalities,
            "minimum_slack_solves": inequalities if level != "fast" else 0,
        }


class _RowBuilder:
    def __init__(self) -> None:
        self.expressions: list[LinearExpression] = []
        self.rhs: list[float] = []
        self.records: list[RowRecord] = []
        self._keys: set[str] = set()
        self.vacuous: list[VacuousConstraint] = []

    def add(
        self,
        expression: LinearExpression,
        rhs: float,
        *,
        family: str,
        qualifiers: Mapping[str, Any],
        sense: RowSense = RowSense.GREATER_EQUAL,
        unit: str = "utility",
        report_factor: float = 1.0,
        report_unit: str | None = None,
        classifiable: bool = True,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        key = format_constraint_id(family, qualifiers)
        if key in self._keys:
            raise RequirementValidationError(
                "requirements generate a duplicate semantic row", context={"row": key}
            )
        self._keys.add(key)
        metadata = _metadata(qualifiers)
        metadata.update(_metadata(extra_metadata or {}))
        linear = expression.without_constant()
        self.expressions.append(linear)
        canonical_rhs = float(rhs) - expression.constant
        self.rhs.append(canonical_rhs)
        self.records.append(
            RowRecord(
                key=key,
                family=family,
                sense=sense,
                scale=_make_scale(
                    linear,
                    canonical_rhs,
                    report_factor=report_factor,
                    original_unit=unit,
                    report_unit=report_unit,
                ),
                classifiable=classifiable and sense is RowSense.GREATER_EQUAL,
                metadata=metadata,
            )
        )

    def skip(
        self,
        *,
        family: str,
        qualifiers: Mapping[str, Any],
        reason: str,
    ) -> None:
        key = format_constraint_id(family, qualifiers)
        if key in self._keys:
            raise RequirementValidationError("duplicate semantic row", context={"row": key})
        self._keys.add(key)
        self.vacuous.append(
            VacuousConstraint(
                key=key,
                family=family,
                reason=reason,
                metadata=tuple(sorted(_metadata(qualifiers).items())),
            )
        )


def _utility_terms(
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    builder: LinearExpressionBuilder,
    *,
    agent_index: int,
    true_index: int,
    report_index: int,
    factor: float,
) -> None:
    for outcome_index, value in enumerate(environment.values[agent_index][true_index]):
        builder.add_term(layout.allocation(report_index, outcome_index), factor * value)
    builder.add_term(layout.payment(report_index, agent_index), -factor)


def _deviation_expression(
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    *,
    agent_index: int,
    true_index: int,
    deviation_index: int,
    factor: float = 1.0,
    builder: LinearExpressionBuilder | None = None,
) -> LinearExpressionBuilder:
    """Accumulate ``factor * (truthful utility - deviation utility)``."""

    expression = builder if builder is not None else LinearExpressionBuilder()
    _utility_terms(
        environment,
        layout,
        expression,
        agent_index=agent_index,
        true_index=true_index,
        report_index=true_index,
        factor=factor,
    )
    _utility_terms(
        environment,
        layout,
        expression,
        agent_index=agent_index,
        true_index=true_index,
        report_index=deviation_index,
        factor=-factor,
    )
    return expression


def _compile_dsic(
    rows: _RowBuilder,
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    requirement: DSIC,
) -> None:
    if not environment.is_private_values:
        raise RequirementValidationError(
            "DSIC requires PrivateValues; use ExPostIC for interdependent values"
        )
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
                    expression = _deviation_expression(
                        environment,
                        layout,
                        agent_index=agent_index,
                        true_index=base + true_position * stride,
                        deviation_index=base + deviation_position * stride,
                    )
                    rows.add(
                        expression.build(),
                        0.0,
                        family="dsic",
                        qualifiers={
                            "agent": agent,
                            "true_type": true_type,
                            "deviation": deviation,
                            "others_report": others,
                        },
                    )


def _compile_ex_post_ic(
    rows: _RowBuilder,
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    requirement: ExPostIC,
) -> None:
    for agent_index, agent in enumerate(environment.agents):
        if not _selected(requirement.agents, agent):
            continue
        stride = environment._strides[agent_index]
        types = environment.types[agent]
        for true_index, profile in enumerate(environment.profiles):
            true_position = environment._type_position(agent_index, true_index)
            true_type = types[true_position]
            if not _selected_for_agent(requirement.true_types, agent, true_type):
                continue
            for deviation_position, deviation in enumerate(types):
                if deviation == true_type or not _selected_for_agent(
                    requirement.deviations, agent, deviation
                ):
                    continue
                expression = _deviation_expression(
                    environment,
                    layout,
                    agent_index=agent_index,
                    true_index=true_index,
                    deviation_index=true_index + (deviation_position - true_position) * stride,
                )
                rows.add(
                    expression.build(),
                    0.0,
                    family="ex_post_ic",
                    qualifiers={
                        "agent": agent,
                        "true_type": true_type,
                        "deviation": deviation,
                        "true_profile": profile,
                    },
                )


def _compile_bic(
    rows: _RowBuilder,
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    requirement: BIC,
) -> None:
    prior = environment.prior
    masses = environment._profile_masses
    if prior is None or masses is None:
        raise EnvironmentValidationError("BIC compilation requires a joint prior")
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
                qualifiers = {
                    "agent": agent,
                    "true_type": true_type,
                    "deviation": deviation,
                }
                if marginal == 0.0:
                    if requirement.zero_probability_types is ZeroProbabilityPolicy.ERROR:
                        raise ZeroProbabilityTypeError(
                            "BIC compilation encountered a zero-marginal type",
                            context={"agent": agent, "true_type": true_type},
                        )
                    if requirement.zero_probability_types is ZeroProbabilityPolicy.VACUOUS:
                        rows.skip(
                            family="bic", qualifiers=qualifiers, reason="zero_probability_type"
                        )
                        continue
                    for others, base in others_entries:
                        expression = _deviation_expression(
                            environment,
                            layout,
                            agent_index=agent_index,
                            true_index=base + true_position * stride,
                            deviation_index=base + deviation_position * stride,
                        )
                        rows.add(
                            expression.build(),
                            0.0,
                            family="bic",
                            qualifiers={**qualifiers, "others_type": others},
                            extra_metadata={
                                "zero_probability_policy": "enforce_ex_post",
                                "conditioning_mass": marginal,
                            },
                        )
                    continue

                expression = LinearExpressionBuilder()
                for _, base in others_entries:
                    true_index = base + true_position * stride
                    probability = masses[true_index]
                    _deviation_expression(
                        environment,
                        layout,
                        agent_index=agent_index,
                        true_index=true_index,
                        deviation_index=base + deviation_position * stride,
                        factor=probability,
                        builder=expression,
                    )
                rows.add(
                    expression.build(),
                    0.0,
                    family="bic",
                    qualifiers=qualifiers,
                    report_factor=1.0 / marginal,
                    report_unit="conditional_utility",
                    extra_metadata={"conditioning_mass": marginal},
                )


def _compile_ex_post_ir(
    rows: _RowBuilder,
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    requirement: ExPostIR,
) -> None:
    for agent_index, agent in enumerate(environment.agents):
        if not _selected(requirement.agents, agent):
            continue
        for profile_index, profile in enumerate(environment.profiles):
            true_type = profile[agent]
            if not _selected_for_agent(requirement.types, agent, true_type):
                continue
            expression = LinearExpressionBuilder()
            _utility_terms(
                environment,
                layout,
                expression,
                agent_index=agent_index,
                true_index=profile_index,
                report_index=profile_index,
                factor=1.0,
            )
            rows.add(
                expression.build(),
                requirement.outside_option.value(agent, profile),
                family="ex_post_ir",
                qualifiers={"agent": agent, "true_profile": profile},
                extra_metadata={"true_type": true_type},
            )


def _compile_interim_ir(
    rows: _RowBuilder,
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    requirement: InterimIR,
) -> None:
    prior = environment.prior
    masses = environment._profile_masses
    if prior is None or masses is None:
        raise EnvironmentValidationError("interim IR compilation requires a joint prior")
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
                        "interim IR compilation encountered a zero-marginal type",
                        context=qualifiers,
                    )
                if requirement.zero_probability_types is ZeroProbabilityPolicy.VACUOUS:
                    rows.skip(
                        family="interim_ir",
                        qualifiers=qualifiers,
                        reason="zero_probability_type",
                    )
                    continue
                for others, base in others_entries:
                    profile_index = base + type_position * stride
                    expression = LinearExpressionBuilder()
                    _utility_terms(
                        environment,
                        layout,
                        expression,
                        agent_index=agent_index,
                        true_index=profile_index,
                        report_index=profile_index,
                        factor=1.0,
                    )
                    rows.add(
                        expression.build(),
                        requirement.outside_option.value(
                            agent, environment.profiles[profile_index]
                        ),
                        family="interim_ir",
                        qualifiers={**qualifiers, "others_type": others},
                        extra_metadata={
                            "zero_probability_policy": "enforce_ex_post",
                            "conditioning_mass": marginal,
                        },
                    )
                continue

            expression = LinearExpressionBuilder()
            outside = 0.0
            for _, base in others_entries:
                profile_index = base + type_position * stride
                probability = masses[profile_index]
                _utility_terms(
                    environment,
                    layout,
                    expression,
                    agent_index=agent_index,
                    true_index=profile_index,
                    report_index=profile_index,
                    factor=probability,
                )
                outside += probability * requirement.outside_option.value(
                    agent, environment.profiles[profile_index]
                )
            rows.add(
                expression.build(),
                outside,
                family="interim_ir",
                qualifiers=qualifiers,
                report_factor=1.0 / marginal,
                report_unit="conditional_utility",
                extra_metadata={"conditioning_mass": marginal},
            )


def _compile_ex_ante_ir(
    rows: _RowBuilder,
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    requirement: ExAnteIR,
) -> None:
    masses = environment._profile_masses
    if environment.prior is None or masses is None:
        raise EnvironmentValidationError("ex-ante IR compilation requires a joint prior")
    for agent_index, agent in enumerate(environment.agents):
        if not _selected(requirement.agents, agent):
            continue
        expression = LinearExpressionBuilder()
        outside = 0.0
        for profile_index, profile in enumerate(environment.profiles):
            probability = masses[profile_index]
            _utility_terms(
                environment,
                layout,
                expression,
                agent_index=agent_index,
                true_index=profile_index,
                report_index=profile_index,
                factor=probability,
            )
            outside += probability * requirement.outside_option.value(agent, profile)
        rows.add(
            expression.build(),
            outside,
            family="ex_ante_ir",
            qualifiers={"agent": agent},
            unit="expected_utility",
        )


def _selected_reports(environment: FiniteEnvironment, reports: Any) -> tuple[ReportProfile, ...]:
    if reports is None:
        return environment.report_profiles
    return tuple(environment.coerce_report_profile(report) for report in reports)


def _compile_budget(
    rows: _RowBuilder,
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    requirement: Any,
) -> None:
    if isinstance(requirement, NoSubsidies):
        for report in _selected_reports(environment, requirement.reports):
            report_index = environment.profile_index(report)
            for agent in environment.agents:
                if not _selected(requirement.agents, agent):
                    continue
                expression = LinearExpression(
                    ((layout.payment(report_index, environment.agent_index(agent)), 1.0),)
                )
                rows.add(
                    expression,
                    0.0,
                    family="no_subsidies",
                    qualifiers={"agent": agent, "report": report},
                    unit="money",
                )
        return
    if isinstance(requirement, PaymentBounds):
        for report in _selected_reports(environment, requirement.reports):
            report_index = environment.profile_index(report)
            for agent in environment.agents:
                if not _selected(requirement.agents, agent):
                    continue
                column = layout.payment(report_index, environment.agent_index(agent))
                if requirement.lower is not None:
                    rows.add(
                        LinearExpression(((column, 1.0),)),
                        requirement.lower,
                        family="payment_bounds",
                        qualifiers={"agent": agent, "report": report, "side": "lower"},
                        unit="money",
                    )
                if requirement.upper is not None:
                    rows.add(
                        LinearExpression(((column, -1.0),)),
                        -requirement.upper,
                        family="payment_bounds",
                        qualifiers={"agent": agent, "report": report, "side": "upper"},
                        unit="money",
                    )
        return
    if isinstance(requirement, (WeakBudgetBalance, StrongBudgetBalance)):
        target = (
            requirement.floor if isinstance(requirement, WeakBudgetBalance) else requirement.target
        )
        sense = (
            RowSense.GREATER_EQUAL if isinstance(requirement, WeakBudgetBalance) else RowSense.EQUAL
        )
        if requirement.timing is BudgetTiming.EX_POST:
            for report in environment.report_profiles:
                report_index = environment.profile_index(report)
                expression = LinearExpression(
                    tuple(
                        (layout.payment(report_index, agent_index), 1.0)
                        for agent_index in range(environment.n_agents)
                    )
                )
                rows.add(
                    expression,
                    target,
                    family=requirement.family,
                    qualifiers={"report": report},
                    sense=sense,
                    unit="money",
                    classifiable=sense is RowSense.GREATER_EQUAL,
                )
        else:
            masses = environment._profile_masses
            if environment.prior is None or masses is None:
                raise EnvironmentValidationError("ex-ante budget balance requires a prior")
            expression_builder = LinearExpressionBuilder()
            for report_index, probability in enumerate(masses):
                for agent_index in range(environment.n_agents):
                    expression_builder.add_term(
                        layout.payment(report_index, agent_index), probability
                    )
            rows.add(
                expression_builder.build(),
                target,
                family=requirement.family,
                qualifiers={"timing": "ex_ante"},
                sense=sense,
                unit="expected_money",
                classifiable=sense is RowSense.GREATER_EQUAL,
            )
        return
    raise UnsupportedRequirementError("unsupported budget requirement")


def _translate_coordinate(
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    coordinate: AllocationCoordinate | PaymentCoordinate,
) -> int:
    report = environment.coerce_report_profile(coordinate.report)
    report_index = environment.profile_index(report)
    if isinstance(coordinate, AllocationCoordinate):
        return layout.allocation(report_index, environment.outcome_index(coordinate.outcome))
    return layout.payment(report_index, environment.agent_index(coordinate.agent))


def _compile_custom(
    rows: _RowBuilder,
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    requirement: CustomLinearConstraint | CustomLinearRequirement,
) -> None:
    constraints = (
        (requirement,)
        if isinstance(requirement, CustomLinearConstraint)
        else requirement.constraints
    )
    for constraint in constraints:
        qualifiers = dict(constraint.spec.id.qualifiers)
        expression = LinearExpressionBuilder(constant=constraint.spec.expression.constant)
        for coordinate, coefficient in constraint.spec.expression.terms:
            expression.add_term(_translate_coordinate(environment, layout, coordinate), coefficient)
        sense = (
            RowSense.EQUAL
            if constraint.spec.relation is ConstraintRelation.EQUAL
            else RowSense.GREATER_EQUAL
        )
        rows.add(
            expression.build(),
            0.0,
            family=constraint.family,
            qualifiers=qualifiers,
            sense=sense,
            unit=constraint.spec.economic_unit,
            classifiable=sense is RowSense.GREATER_EQUAL,
        )


def _compile_capacity(
    rows: _RowBuilder,
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    requirement: Capacity,
) -> None:
    usage = {environment.outcome_index(outcome): amount for outcome, amount in requirement.usage}
    reports = (
        environment.report_profiles
        if requirement.profiles is None
        else tuple(environment.coerce_report_profile(item) for item in requirement.profiles)
    )
    for report in reports:
        report_index = environment.profile_index(report)
        upper = LinearExpression(
            tuple(
                (layout.allocation(report_index, outcome_index), -amount)
                for outcome_index, amount in usage.items()
            )
        )
        rows.add(
            upper,
            -requirement.upper,
            family="capacity",
            qualifiers={"resource": requirement.resource, "report": report, "side": "upper"},
            unit="resource",
        )
        if requirement.lower is not None:
            rows.add(
                -upper,
                requirement.lower,
                family="capacity",
                qualifiers={
                    "resource": requirement.resource,
                    "report": report,
                    "side": "lower",
                },
                unit="resource",
            )


def _compile_symmetry(
    rows: _RowBuilder,
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    requirement: Symmetry,
) -> None:
    for operation in requirement.operations:
        agent_map = dict(operation.agents)
        type_maps = {agent: dict(mapping) for agent, mapping in operation.types}
        outcome_map = dict(operation.outcomes)
        if set(agent_map) != set(environment.agents) or set(agent_map.values()) != set(
            environment.agents
        ):
            raise RequirementValidationError(
                "symmetry agent action must be a permutation of all agents"
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
            source_index = environment.profile_index(report)
            target_index = environment.profile_index(transformed)
            for outcome in environment.outcomes:
                transformed_outcome = outcome_map[outcome]
                expression = LinearExpression(
                    (
                        (layout.allocation(source_index, environment.outcome_index(outcome)), 1.0),
                        (
                            layout.allocation(
                                target_index,
                                environment.outcome_index(transformed_outcome),
                            ),
                            -1.0,
                        ),
                    )
                )
                rows.add(
                    expression,
                    0.0,
                    family="symmetry",
                    qualifiers={
                        "operation": operation.name,
                        "report": report,
                        "outcome": outcome,
                        "component": "allocation",
                    },
                    sense=RowSense.EQUAL,
                    unit="probability",
                    classifiable=False,
                )
            for source in environment.agents:
                target = agent_map[source]
                expression = LinearExpression(
                    (
                        (layout.payment(source_index, environment.agent_index(source)), 1.0),
                        (layout.payment(target_index, environment.agent_index(target)), -1.0),
                    )
                )
                rows.add(
                    expression,
                    0.0,
                    family="symmetry",
                    qualifiers={
                        "operation": operation.name,
                        "report": report,
                        "agent": source,
                        "component": "payment",
                    },
                    sense=RowSense.EQUAL,
                    unit="money",
                    classifiable=False,
                )


def _agent_weights(environment: FiniteEnvironment, values: Any) -> dict[AgentId, float]:
    if values is None:
        return {agent: 1.0 for agent in environment.agents}
    weights = dict(values)
    unknown = set(weights) - set(environment.agents)
    if unknown:
        raise EnvironmentValidationError(
            "objective weights reference unknown agents", context={"agents": unknown}
        )
    return {agent: float(weights.get(agent, 0.0)) for agent in environment.agents}


def _objective_expression(
    environment: FiniteEnvironment,
    layout: MechanismVariableLayout,
    objective: LinearObjective,
) -> LinearExpression:
    if isinstance(objective, LinearCombination):
        builder = LinearExpressionBuilder(constant=objective.constant)
        for coefficient, term in objective.terms:
            builder.add_expression(
                _objective_expression(environment, layout, term), factor=coefficient
            )
        return builder.build()
    if isinstance(objective, CustomLinearObjective):
        builder = LinearExpressionBuilder(constant=objective.expression.constant)
        for coordinate, coefficient in objective.expression.terms:
            builder.add_term(_translate_coordinate(environment, layout, coordinate), coefficient)
        return builder.build()

    masses = environment._profile_masses
    if environment.prior is None or masses is None:
        raise EnvironmentValidationError("expected designer objectives require a joint prior")
    if isinstance(objective, ExpectedRevenue):
        weights = _agent_weights(environment, objective.agent_weights)
        builder = LinearExpressionBuilder()
        for report_index, probability in enumerate(masses):
            for agent in environment.agents:
                builder.add_term(
                    layout.payment(report_index, environment.agent_index(agent)),
                    probability * weights[agent],
                )
        return builder.build()
    if isinstance(
        objective,
        (ExpectedSocialWelfare, ExpectedResidualSurplus, WeightedAgentUtilities),
    ):
        raw_weights: Any
        if isinstance(objective, WeightedAgentUtilities):
            raw_weights = objective.weights
        elif isinstance(objective, ExpectedSocialWelfare):
            raw_weights = objective.agent_weights
        else:
            raw_weights = None
        weights = _agent_weights(environment, raw_weights)
        builder = LinearExpressionBuilder()
        for profile_index, probability in enumerate(masses):
            for outcome_index in range(environment.n_outcomes):
                welfare = math.fsum(
                    weights[agent]
                    * environment.values[environment.agent_index(agent)][profile_index][
                        outcome_index
                    ]
                    for agent in environment.agents
                )
                builder.add_term(
                    layout.allocation(profile_index, outcome_index),
                    probability * welfare,
                )
            if isinstance(objective, (ExpectedResidualSurplus, WeightedAgentUtilities)):
                for agent in environment.agents:
                    builder.add_term(
                        layout.payment(profile_index, environment.agent_index(agent)),
                        -probability * weights[agent],
                    )
        return builder.build()
    raise UnsupportedRequirementError(f"unsupported designer objective: {type(objective).__name__}")


def compile_design_problem(
    environment: FiniteEnvironment,
    objective: LinearObjective,
    requirements: Iterable[Any],
) -> CompiledDesign:
    """Compile a randomized finite direct-mechanism problem once."""

    if not isinstance(environment, FiniteEnvironment):
        raise TypeError("environment must be a FiniteEnvironment")
    if not isinstance(objective, LinearObjective):
        raise TypeError("objective must be a LinearObjective")
    declared = tuple(requirements)
    layout = MechanismVariableLayout.create(environment)
    rows = _RowBuilder()

    # Allocation normalization is an accounting equality, not an economic row
    # to classify as active over the optimal face.
    for report in environment.report_profiles:
        report_index = environment.profile_index(report)
        rows.add(
            LinearExpression(
                tuple(
                    (layout.allocation(report_index, outcome_index), 1.0)
                    for outcome_index in range(environment.n_outcomes)
                )
            ),
            1.0,
            family="allocation_normalization",
            qualifiers={"report": report},
            sense=RowSense.EQUAL,
            unit="probability",
            classifiable=False,
        )

    for requirement in declared:
        if isinstance(requirement, DSIC):
            _compile_dsic(rows, environment, layout, requirement)
        elif isinstance(requirement, ExPostIC):
            _compile_ex_post_ic(rows, environment, layout, requirement)
        elif isinstance(requirement, BIC):
            _compile_bic(rows, environment, layout, requirement)
        elif isinstance(requirement, ExPostIR):
            _compile_ex_post_ir(rows, environment, layout, requirement)
        elif isinstance(requirement, InterimIR):
            _compile_interim_ir(rows, environment, layout, requirement)
        elif isinstance(requirement, ExAnteIR):
            _compile_ex_ante_ir(rows, environment, layout, requirement)
        elif isinstance(
            requirement,
            (NoSubsidies, PaymentBounds, WeakBudgetBalance, StrongBudgetBalance),
        ):
            _compile_budget(rows, environment, layout, requirement)
        elif isinstance(requirement, (CustomLinearConstraint, CustomLinearRequirement)):
            _compile_custom(rows, environment, layout, requirement)
        elif isinstance(requirement, Capacity):
            _compile_capacity(rows, environment, layout, requirement)
        elif isinstance(requirement, Symmetry):
            _compile_symmetry(rows, environment, layout, requirement)
        else:
            raise UnsupportedRequirementError(
                f"unsupported design requirement: {type(requirement).__name__}"
            )

    objective_expression = _objective_expression(environment, layout, objective)
    objective_vector = np.zeros(layout.n_columns, dtype=float)
    for column, coefficient in objective_expression.terms:
        objective_vector[column] = coefficient

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    for row_index, expression in enumerate(rows.expressions):
        for column, coefficient in expression.terms:
            matrix_rows.append(row_index)
            matrix_columns.append(column)
            matrix_values.append(coefficient)
    matrix = sparse.coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(rows.records), layout.n_columns),
        dtype=float,
    ).tocsr()

    lower = np.full(layout.n_columns, -np.inf, dtype=float)
    upper = np.full(layout.n_columns, np.inf, dtype=float)
    lower[layout.allocation_slice] = 0.0
    upper[layout.allocation_slice] = 1.0

    columns: list[ColumnRecord] = []
    for key in layout.index:
        if key.block == "allocation":
            report_index, outcome_index = key.coordinates
            columns.append(
                ColumnRecord(
                    key=str(key),
                    block="allocation",
                    metadata={
                        "report_index": report_index,
                        "outcome_index": outcome_index,
                        "report": repr(environment.report_profiles[report_index]),
                        "outcome": repr(environment.outcomes[outcome_index]),
                    },
                )
            )
        else:
            report_index, agent_index = key.coordinates
            columns.append(
                ColumnRecord(
                    key=str(key),
                    block="payment",
                    metadata={
                        "report_index": report_index,
                        "agent_index": agent_index,
                        "report": repr(environment.report_profiles[report_index]),
                        "agent": repr(environment.agents[agent_index]),
                    },
                )
            )

    model = CompiledLP(
        matrix,
        np.asarray(rows.rhs, dtype=float),
        objective_vector,
        objective_offset=objective_expression.constant,
        lower_bounds=lower,
        upper_bounds=upper,
        rows=tuple(rows.records),
        columns=tuple(columns),
    )
    return CompiledDesign(
        environment=environment,
        objective=objective,
        requirements=declared,
        layout=layout,
        model=model,
        vacuous_constraints=tuple(rows.vacuous),
    )
