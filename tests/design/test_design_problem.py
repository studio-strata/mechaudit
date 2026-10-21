"""End-to-end design, failure, and candidate-assessment tests."""

from __future__ import annotations

import pytest

from mechaudit.core.environment import FiniteEnvironment
from mechaudit.core.valuations import PrivateValues
from mechaudit.design.problem import DesignProblem
from mechaudit.exceptions import InfeasibleDesignError, UnboundedDesignError
from mechaudit.mechanisms.direct import DirectMechanism
from mechaudit.objectives import (
    CustomLinearObjective,
    ExpectedRevenue,
    ExpectedSocialWelfare,
)
from mechaudit.requirements.base import AllocationCoordinate, PaymentCoordinate
from mechaudit.requirements.custom import CustomLinearConstraint
from mechaudit.requirements.feasibility import Capacity
from mechaudit.requirements.incentive import DSIC
from mechaudit.requirements.participation import ExPostIR


def _one_buyer_environment(*, low_value: float = 1.0, high_value: float = 2.0) -> FiniteEnvironment:
    return FiniteEnvironment(
        agents={"buyer": ["low", "high"]},
        outcomes=["no_sale", "sale"],
        prior={"low": 0.5, "high": 0.5},
        values=PrivateValues(
            lambda _agent, type_value, outcome: (
                {"low": low_value, "high": high_value}[type_value] if outcome == "sale" else 0.0
            )
        ),
    )


def _revenue_problem(environment: FiniteEnvironment) -> DesignProblem:
    return DesignProblem(
        environment=environment,
        objective=ExpectedRevenue(),
        requirements=(DSIC(), ExPostIR()),
    )


def test_unsupported_design_configuration_has_actionable_errors() -> None:
    environment = _one_buyer_environment()

    with pytest.raises(NotImplementedError, match="randomized=False is not supported"):
        environment.design(objective=ExpectedRevenue(), randomized=False)

    with pytest.raises(ValueError, match="expected 'highs' or an LPBackend instance"):
        _revenue_problem(environment).solve(backend="unsupported")


def test_revenue_equivalent_optimal_face_has_value_one() -> None:
    environment = _one_buyer_environment()
    solution = _revenue_problem(environment).solve()

    assert solution.objective_value == pytest.approx(1.0, abs=1e-9)
    assert solution.audit.passed
    high = environment.profile_index("high")
    sale = environment.outcome_index("sale")
    assert solution.mechanism.allocation[high, sale] == pytest.approx(1.0, abs=1e-8)


def test_values_one_and_three_have_unique_myerson_solution() -> None:
    environment = _one_buyer_environment(high_value=3.0)
    solution = _revenue_problem(environment).solve()
    low = environment.profile_index("low")
    high = environment.profile_index("high")
    sale = environment.outcome_index("sale")
    buyer = environment.agent_index("buyer")

    assert solution.objective_value == pytest.approx(1.5, abs=1e-9)
    assert solution.mechanism.allocation[low, sale] == pytest.approx(0.0, abs=1e-8)
    assert solution.mechanism.allocation[high, sale] == pytest.approx(1.0, abs=1e-8)
    assert solution.mechanism.payments[low, buyer] == pytest.approx(0.0, abs=1e-8)
    assert solution.mechanism.payments[high, buyer] == pytest.approx(3.0, abs=1e-8)


def test_expected_welfare_allocates_to_every_positive_type() -> None:
    environment = _one_buyer_environment(high_value=3.0)
    problem = DesignProblem(
        environment=environment,
        objective=ExpectedSocialWelfare(),
        requirements=(DSIC(), ExPostIR()),
    )
    solution = problem.solve()
    sale = environment.outcome_index("sale")

    assert solution.objective_value == pytest.approx(2.0, abs=1e-9)
    assert all(
        solution.mechanism.allocation[environment.profile_index(type_value), sale]
        == pytest.approx(1.0, abs=1e-8)
        for type_value in ("low", "high")
    )
    assert solution.audit.passed


def test_unbounded_revenue_diagnostic_and_ir_bounded_variant() -> None:
    environment = FiniteEnvironment(
        agents={"buyer": ["only"]},
        outcomes=["no_sale", "sale"],
        prior={"only": 1.0},
        values=PrivateValues(lambda _agent, _type, outcome: 1.0 if outcome == "sale" else 0.0),
    )
    unbounded = DesignProblem(environment=environment, objective=ExpectedRevenue())

    with pytest.raises(UnboundedDesignError, match="IR condition or payment cap"):
        unbounded.solve()

    bounded = unbounded.with_requirements(ExPostIR())
    solution = bounded.solve()
    assert solution.objective_value == pytest.approx(1.0, abs=1e-9)
    assert solution.audit.passed


def test_infeasible_custom_payment_constraints_raise_domain_error() -> None:
    environment = FiniteEnvironment(
        agents={"buyer": ["only"]},
        outcomes=["outcome"],
        prior={"only": 1.0},
        values=PrivateValues(lambda _agent, _type, _outcome: 0.0),
    )
    report = environment.report_profiles[0]
    payment = PaymentCoordinate(report, "buyer")
    requirements = (
        CustomLinearConstraint("minimum", {payment: 1.0}, sense=">=", rhs=1.0),
        CustomLinearConstraint("maximum", {payment: 1.0}, sense="<=", rhs=0.0),
    )
    problem = DesignProblem(
        environment=environment,
        objective=ExpectedRevenue(),
        requirements=requirements,
    )

    with pytest.raises(InfeasibleDesignError, match="infeasible"):
        problem.solve()


def test_custom_objective_constraint_and_capacity_are_compiled_and_audited() -> None:
    environment = FiniteEnvironment(
        agents={"buyer": ["only"]},
        outcomes=["off", "on"],
        prior={"only": 1.0},
        values=PrivateValues(lambda _agent, _type, outcome: 10.0 if outcome == "on" else 0.0),
    )
    report = environment.report_profiles[0]
    on = AllocationCoordinate(report, "on")
    custom = CustomLinearConstraint(
        "policy_cap", {on: 1.0}, sense="<=", rhs=0.4, economic_unit="probability"
    )
    custom_problem = DesignProblem(
        environment=environment,
        objective=CustomLinearObjective("on_probability", {on: 1.0}),
        requirements=(custom,),
    )
    custom_solution = custom_problem.solve()

    assert custom_solution.objective_value == pytest.approx(0.4, abs=1e-9)
    assert custom_solution.audit.passed
    assert custom_solution.audit.section("custom", {"custom"}).minimum_slack == pytest.approx(
        0.0, abs=1e-9
    )

    capacity_problem = DesignProblem(
        environment=environment,
        objective=ExpectedSocialWelfare(),
        requirements=(Capacity(resource="resource", usage={"off": 0, "on": 1}, upper=0.25),),
    )
    capacity_solution = capacity_problem.solve()
    assert capacity_solution.objective_value == pytest.approx(2.5, abs=1e-9)
    assert capacity_solution.mechanism.allocation[
        0, environment.outcome_index("on")
    ] == pytest.approx(0.25, abs=1e-9)
    assert capacity_solution.audit.passed


def test_candidate_assessment_can_skip_optimal_face_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _one_buyer_environment()
    problem = _revenue_problem(environment)
    solution = problem.solve()
    pooling = DirectMechanism(
        allocation={
            "low": {"no_sale": 0.0, "sale": 1.0},
            "high": {"no_sale": 0.0, "sale": 1.0},
        },
        payments={"low": 1.0, "high": 1.0},
        environment=environment,
    )

    def unexpected_face_analysis(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("optimal-face analysis should not run")

    monkeypatch.setattr(type(solution), "analyze_optima", unexpected_face_analysis)
    assessment = problem.assess(
        pooling,
        solution=solution,
        analyze_optimal_face=False,
    )

    assert assessment.is_feasible
    assert assessment.is_optimal
    assert assessment.objective_value == pytest.approx(1.0, abs=1e-9)
    assert assessment.optimality_gap == pytest.approx(0.0, abs=1e-9)
    assert assessment.universally_active_constraints == ()
    assert assessment.active_but_nonuniversal_constraints == assessment.active_constraints


def test_audit_activity_and_optimal_face_share_constraint_ids() -> None:
    from mechaudit.optimality.essentiality import constraint_cost
    from mechaudit.requirements.budget import NoSubsidies

    environment = _one_buyer_environment()
    problem = DesignProblem(
        environment=environment,
        objective=ExpectedRevenue(),
        requirements=(DSIC(), ExPostIR(), NoSubsidies()),
    )
    solution = problem.solve()
    audit_ids = {check.constraint_id for check in solution.audit.checks}
    activity_ids = {item.constraint_id for item in solution.selected_activity}
    face = solution.analyze_optima(level="fast")

    assert activity_ids
    assert activity_ids <= audit_ids
    assert {record.constraint_id for record in face.records} <= audit_ids
    assert "no_subsidies[agent='buyer', report={'buyer': 'high'}]" in activity_ids
    # An identifier read from the audit report addresses the same compiled row.
    worst = solution.audit.ic.worst_deviations(1)["constraint_id"].iloc[0]
    assert constraint_cost(solution, worst).constraint_ids == (worst,)


def test_assessment_uses_solution_tolerances_and_excludes_violated_rows() -> None:
    from mechaudit.options import SolveOptions, TolerancePolicy

    environment = _one_buyer_environment()
    problem = _revenue_problem(environment)
    tolerances = TolerancePolicy(feasibility=1e-6)
    solution = problem.solve(options=SolveOptions(tolerances=tolerances))
    # Overcharge the high type by 0.5: its IR row is violated, not active.
    candidate = DirectMechanism(
        allocation={"low": {"no_sale": 0.0, "sale": 1.0}, "high": {"no_sale": 0.0, "sale": 1.0}},
        payments={"low": 1.0, "high": 2.5},
        environment=environment,
    )

    assessment = problem.assess(candidate, solution=solution, analyze_optimal_face=False)

    assert assessment.audit.feasibility_tolerance == pytest.approx(1e-6)
    assert not assessment.is_feasible
    violated = {check.constraint_id for check in assessment.audit.checks if not check.passed}
    assert violated
    assert not violated.intersection(assessment.active_constraints)
