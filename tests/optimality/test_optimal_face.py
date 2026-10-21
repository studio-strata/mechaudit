from __future__ import annotations

import json

import pytest

from mechaudit import (
    DSIC,
    AllocationCoordinate,
    CustomLinearConstraint,
    CustomLinearObjective,
    ExpectedRevenue,
    ExPostIR,
    FiniteEnvironment,
    NoSubsidies,
    PaymentBounds,
    PaymentCoordinate,
    PrivateValues,
)
from mechaudit.optimality import ActivityClassification, DualRelevance


def _revenue_equivalent_solution():
    environment = FiniteEnvironment(
        agents={"buyer": ("low", "high")},
        outcomes=("no_sale", "sale"),
        prior={"low": 0.5, "high": 0.5},
        values=PrivateValues(
            lambda _agent, own_type, outcome: (
                {"low": 1.0, "high": 2.0}[own_type] if outcome == "sale" else 0.0
            )
        ),
    )
    problem = environment.design(
        objective=ExpectedRevenue(),
        requirements=(DSIC(), ExPostIR(), NoSubsidies()),
    )
    return problem.solve()


def _find(report, *, family: str, text: str):
    return next(
        record
        for record in report.constraints
        if record.family == family and text in record.constraint_id
    )


def test_full_exact_face_recovers_elevator_pitch_partition() -> None:
    solution = _revenue_equivalent_solution()
    face = solution.analyze_optima(level="full", objective_delta=0.0)

    low_to_high = _find(face, family="dsic", text="true_type='low'")
    high_to_low = _find(face, family="dsic", text="true_type='high'")
    low_ir = _find(face, family="ex_post_ir", text="'low'")
    high_ir = _find(face, family="ex_post_ir", text="'high'")
    low_payment = _find(face, family="no_subsidies", text="'low'")
    high_payment = _find(face, family="no_subsidies", text="'high'")

    assert high_to_low.classification is ActivityClassification.UNIVERSALLY_ACTIVE
    assert low_ir.classification is ActivityClassification.UNIVERSALLY_ACTIVE
    assert low_to_high.classification is ActivityClassification.SOMETIMES_ACTIVE
    assert high_ir.classification is ActivityClassification.SOMETIMES_ACTIVE
    assert low_payment.classification is ActivityClassification.SOMETIMES_ACTIVE
    assert high_payment.classification is ActivityClassification.NEVER_ACTIVE
    assert high_payment.minimum_slack == pytest.approx(1.0)
    assert not face.inconclusive


def test_wide_delta_rechecks_rows_that_are_universal_only_on_exact_face() -> None:
    solution = _revenue_equivalent_solution()
    exact = solution.analyze_optima(level="full", objective_delta=0.0)
    wide = solution.analyze_optima(level="full", objective_delta=0.25)

    exact_high_ic = _find(exact, family="dsic", text="true_type='high'")
    exact_low_ir = _find(exact, family="ex_post_ir", text="'low'")
    wide_high_ic = _find(wide, family="dsic", text="true_type='high'")
    wide_low_ir = _find(wide, family="ex_post_ir", text="'low'")

    assert exact_high_ic.classification is ActivityClassification.UNIVERSALLY_ACTIVE
    assert exact_low_ir.classification is ActivityClassification.UNIVERSALLY_ACTIVE
    assert wide_high_ic.classification is ActivityClassification.SOMETIMES_ACTIVE
    assert wide_low_ir.classification is ActivityClassification.SOMETIMES_ACTIVE
    assert wide_high_ic.maximum_slack == pytest.approx(0.5)
    assert wide_low_ir.maximum_slack == pytest.approx(0.25)

    payload = json.loads(wide.to_json())
    assert payload["requested_delta"] == pytest.approx(0.25)
    assert payload["effective_delta"] == pytest.approx(0.25)
    assert "optimal_value" in payload["near_optimal_face"]
    assert "delta" in payload["near_optimal_face"]


def test_minimum_slack_distinguishes_sometimes_from_never() -> None:
    environment = FiniteEnvironment(
        agents={"agent": ("only",)},
        outcomes=("outcome",),
        prior={"only": 1.0},
        values=PrivateValues(lambda _agent, _type, _outcome: 0.0),
    )
    report = environment.report_profiles[0]
    payment = PaymentCoordinate(report, "agent")
    problem = environment.design(
        objective=CustomLinearObjective("zero", {}),
        requirements=(
            PaymentBounds(lower=0.0, upper=1.0),
            CustomLinearConstraint("loose_floor", {payment: 1.0}, sense=">=", rhs=-1.0),
        ),
    )
    face = problem.solve().analyze_optima(level="full", objective_delta=0.0)

    bounds = tuple(record for record in face.constraints if record.family == "payment_bounds")
    loose = next(record for record in face.constraints if record.family == "custom")

    assert len(bounds) == 2
    assert all(
        record.classification is ActivityClassification.SOMETIMES_ACTIVE for record in bounds
    )
    assert loose.classification is ActivityClassification.NEVER_ACTIVE
    assert loose.minimum_slack == pytest.approx(1.0)
    assert loose.maximum_slack == pytest.approx(2.0)
    assert loose.candidate_dual == pytest.approx(0.0, abs=1e-8)


def test_fast_mode_does_not_overstate_three_way_classification() -> None:
    solution = _revenue_equivalent_solution()
    face = solution.analyze_optima(level="fast", objective_delta=0.0)

    assert {record.classification for record in face.constraints} <= {
        ActivityClassification.UNIVERSALLY_ACTIVE,
        ActivityClassification.NONUNIVERSAL,
        ActivityClassification.INCONCLUSIVE,
    }
    assert not face.sometimes_active
    assert not face.never_active


def test_dual_redundancy_cost_and_family_ablation_are_distinct_axes() -> None:
    solution = _revenue_equivalent_solution()
    face = solution.analyze_optima(level="full", objective_delta=0.0)
    high_to_low = _find(face, family="dsic", text="true_type='high'")

    dual = face.dual_range(high_to_low)
    redundancy = face.redundancy(high_to_low)
    cost = face.constraint_cost(high_to_low)
    ablation = face.ablate_family("no_subsidies")

    assert dual.relevance is DualRelevance.UNIVERSALLY_PRICED
    assert dual.lower == pytest.approx(0.5)
    assert dual.upper == pytest.approx(0.5)
    assert redundancy.redundant is False
    assert redundancy.minimum_reported_slack == pytest.approx(-2.0)
    assert cost.objective_essential is True
    assert cost.objective_gain == pytest.approx(0.5)
    assert ablation.objective_essential is False
    assert ablation.cost.implementation is not None
    assert ablation.cost.implementation.essential is False


def test_objective_irrelevant_row_can_be_implementation_essential() -> None:
    environment = FiniteEnvironment(
        agents={"agent": ("only",)},
        outcomes=("off", "on"),
        prior={"only": 1.0},
        values=PrivateValues(lambda _agent, _type, _outcome: 0.0),
    )
    report = environment.report_profiles[0]
    on = AllocationCoordinate(report, "on")
    problem = environment.design(
        objective=CustomLinearObjective("on_probability", {on: 1.0}),
        requirements=(PaymentBounds(lower=0.0),),
    )
    face = problem.solve().analyze_optima(level="full", objective_delta=0.0)
    payment_floor = face.constraints[0]

    cost = face.constraint_cost(payment_floor)

    assert cost.objective_essential is False
    assert cost.objective_gain == pytest.approx(0.0)
    assert cost.implementation is not None
    assert cost.implementation.essential is True
    assert cost.implementation.maximum_violation == float("inf")
