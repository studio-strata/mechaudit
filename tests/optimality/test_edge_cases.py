"""Focused optimality edge cases at the solver/report boundary."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mechaudit import (
    CustomLinearConstraint,
    CustomLinearObjective,
    FiniteEnvironment,
    PaymentCoordinate,
    PrivateValues,
)
from mechaudit.design.sparse_model import CompiledLP, RowScale
from mechaudit.optimality import ActivityClassification, AuxiliaryStatus


def _one_payment_environment() -> tuple[FiniteEnvironment, PaymentCoordinate]:
    environment = FiniteEnvironment(
        agents={"agent": ("only",)},
        outcomes=("outcome",),
        prior={"only": 1.0},
        values=PrivateValues(lambda _agent, _type, _outcome: 0.0),
    )
    coordinate = PaymentCoordinate(environment.report_profiles[0], "agent")
    return environment, coordinate


def test_duplicate_active_rows_are_individually_redundant_but_family_is_essential() -> None:
    environment, payment = _one_payment_environment()
    problem = environment.design(
        objective=CustomLinearObjective("payment", {payment: 1.0}),
        requirements=(
            CustomLinearConstraint(
                "cap_a", {payment: 1.0}, sense="<=", rhs=1.0, family="duplicate_cap"
            ),
            CustomLinearConstraint(
                "cap_b", {payment: 1.0}, sense="<=", rhs=1.0, family="duplicate_cap"
            ),
        ),
    )
    face = problem.solve().analyze_optima(level="full", objective_delta=0.0)

    duplicate_rows = tuple(
        record for record in face.constraints if record.family == "duplicate_cap"
    )
    assert len(duplicate_rows) == 2
    assert all(
        record.classification is ActivityClassification.UNIVERSALLY_ACTIVE
        for record in duplicate_rows
    )
    assert all(face.redundancy(record).redundant is True for record in duplicate_rows)

    family = face.ablate_family("duplicate_cap")
    assert family.objective_essential is True
    assert family.objective_gain == float("inf")
    assert family.cost.status is AuxiliaryStatus.UNBOUNDED


def test_individual_constraint_removal_can_make_the_design_unbounded() -> None:
    environment, payment = _one_payment_environment()
    problem = environment.design(
        objective=CustomLinearObjective("payment", {payment: 1.0}),
        requirements=(
            CustomLinearConstraint(
                "only_cap", {payment: 1.0}, sense="<=", rhs=1.0, family="sole_cap"
            ),
        ),
    )
    face = problem.solve().analyze_optima(level="fast", objective_delta=0.0)
    cap = face.constraints[0]

    cost = face.constraint_cost(cap)
    assert cost.status is AuxiliaryStatus.UNBOUNDED
    assert cost.relaxed_objective == float("inf")
    assert cost.objective_gain == float("inf")
    assert cost.objective_essential is True
    assert cost.implementation is None


@pytest.mark.parametrize("offset", (0.0, 7.25))
def test_zero_linear_objective_and_objective_offset_define_a_valid_face(offset: float) -> None:
    environment, payment = _one_payment_environment()
    problem = environment.design(
        objective=CustomLinearObjective("constant", {}, constant=offset),
        requirements=(
            CustomLinearConstraint("floor", {payment: 1.0}, sense=">=", rhs=0.0),
            CustomLinearConstraint("cap", {payment: 1.0}, sense="<=", rhs=1.0),
        ),
    )
    solution = problem.solve()
    face = solution.analyze_optima(level="full", objective_delta=0.0)

    assert solution.objective_value == pytest.approx(offset)
    assert face.optimal_value == pytest.approx(offset)
    assert face.delta == 0.0
    assert face.requested_delta == 0.0
    assert not face.failures
    assert all(
        record.classification is ActivityClassification.SOMETIMES_ACTIVE
        for record in face.constraints
    )
    assert all(record.minimum_slack == pytest.approx(0.0) for record in face.constraints)
    assert all(record.maximum_slack == pytest.approx(1.0) for record in face.constraints)


def _with_row_units(solution, *, solver_factor: float, report_factor: float):
    model = solution.compiled.model
    rows = tuple(
        replace(
            row,
            scale=RowScale(
                factor=row.scale.factor * solver_factor,
                report_factor=row.scale.report_factor * report_factor,
                original_unit=row.scale.original_unit,
                report_unit=row.scale.report_unit,
            ),
        )
        if row.classifiable
        else row
        for row in model.rows
    )
    transformed_model = CompiledLP(
        model.matrix,
        model.rhs,
        model.objective,
        objective_offset=model.objective_offset,
        lower_bounds=model.lower_bounds,
        upper_bounds=model.upper_bounds,
        rows=rows,
        columns=model.columns,
    )
    return replace(solution, compiled=replace(solution.compiled, model=transformed_model))


def test_solver_row_scaling_is_invariant_and_report_units_transform_covariantly() -> None:
    environment, payment = _one_payment_environment()
    problem = environment.design(
        objective=CustomLinearObjective("payment", {payment: 1.0}),
        requirements=(
            CustomLinearConstraint("floor", {payment: 1.0}, sense=">=", rhs=0.0, family="floor"),
            CustomLinearConstraint("cap", {payment: 1.0}, sense="<=", rhs=1.0, family="cap"),
        ),
    )
    solution = problem.solve()
    solver_scaled = _with_row_units(solution, solver_factor=128.0, report_factor=1.0)
    report_scaled = _with_row_units(solution, solver_factor=128.0, report_factor=10.0)

    base = solution.analyze_optima(level="full", objective_delta=0.0)
    solver_report = solver_scaled.analyze_optima(level="full", objective_delta=0.0)
    unit_report = report_scaled.analyze_optima(level="full", objective_delta=0.0)
    base_rows = {record.family: record for record in base.constraints}
    solver_rows = {record.family: record for record in solver_report.constraints}
    unit_rows = {record.family: record for record in unit_report.constraints}

    assert {family: record.classification for family, record in solver_rows.items()} == {
        family: record.classification for family, record in base_rows.items()
    }
    for family in base_rows:
        assert solver_rows[family].selected_slack == pytest.approx(base_rows[family].selected_slack)
        assert solver_rows[family].minimum_slack == pytest.approx(base_rows[family].minimum_slack)
        assert solver_rows[family].maximum_slack == pytest.approx(base_rows[family].maximum_slack)

    assert unit_rows["floor"].classification is ActivityClassification.NEVER_ACTIVE
    assert unit_rows["floor"].selected_slack == pytest.approx(
        10.0 * base_rows["floor"].selected_slack
    )
    assert unit_rows["floor"].minimum_slack == pytest.approx(
        10.0 * base_rows["floor"].minimum_slack
    )
    assert unit_rows["floor"].maximum_slack == pytest.approx(
        10.0 * base_rows["floor"].maximum_slack
    )
    assert unit_rows["cap"].classification is ActivityClassification.UNIVERSALLY_ACTIVE
    assert unit_rows["cap"].candidate_dual == pytest.approx(base_rows["cap"].candidate_dual / 10.0)


def test_equality_removal_is_explicitly_inconclusive_and_not_activity() -> None:
    environment, payment = _one_payment_environment()
    problem = environment.design(
        objective=CustomLinearObjective("zero", {}),
        requirements=(
            CustomLinearConstraint(
                "fixed_payment",
                {payment: 1.0},
                sense="==",
                rhs=0.0,
                family="payment_identity",
            ),
        ),
    )
    solution = problem.solve()
    equality_index = next(
        index
        for index, row in enumerate(solution.compiled.model.rows)
        if row.family == "payment_identity"
    )
    equality_id = solution.compiled.model.rows[equality_index].key
    face = solution.analyze_optima(level="full", objective_delta=0.0)

    assert equality_id not in {record.constraint_id for record in face.constraints}
    with pytest.raises(KeyError):
        face.activity_for(equality_id)

    cost = face.constraint_cost(equality_index)
    assert cost.objective_essential is False
    assert cost.implementation is not None
    assert cost.implementation.status is AuxiliaryStatus.NOT_RUN
    assert cost.implementation.essential is None
    assert cost.implementation.method == "not_implemented_equality_removal"

    direct = face.implementation_essential(equality_index)
    assert direct.status is AuxiliaryStatus.NOT_RUN
    assert direct.essential is None
    assert direct.method == "not_implemented_equality_removal"
