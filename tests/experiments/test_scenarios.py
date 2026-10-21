"""Scenario transformation, prior perturbation, and failure tests."""

from __future__ import annotations

import json

import pytest

from mechaudit.core.environment import FiniteEnvironment
from mechaudit.core.valuations import PrivateValues
from mechaudit.design.problem import DesignProblem
from mechaudit.experiments.scenarios import compare_scenarios
from mechaudit.objectives import ExpectedRevenue
from mechaudit.requirements.budget import NoSubsidies, PaymentBounds
from mechaudit.requirements.incentive import DSIC
from mechaudit.requirements.participation import ExPostIR, InterimIR


def _base_problem() -> DesignProblem:
    environment = FiniteEnvironment(
        agents={"buyer": ["low", "high"]},
        outcomes=["no_sale", "sale"],
        prior={"low": 0.5, "high": 0.5},
        values=PrivateValues(
            lambda _agent, type_value, outcome: (
                {"low": 1.0, "high": 3.0}[type_value] if outcome == "sale" else 0.0
            )
        ),
    )
    return DesignProblem(
        environment=environment,
        objective=ExpectedRevenue(),
        requirements=(DSIC(), ExPostIR()),
    )


def test_scenarios_add_replace_and_prior_perturbation() -> None:
    problem = _base_problem()
    comparison = compare_scenarios(
        problem,
        {
            "baseline": {},
            "add_no_subsidies": {"add": [NoSubsidies()]},
            "interim_ir": {"replace": [(ExPostIR(), InterimIR())]},
            "low_type_likely": {"prior": {"low": 0.9, "high": 0.1}},
        },
    )

    assert [result.status for result in comparison.results] == [
        "optimal",
        "optimal",
        "optimal",
        "optimal",
    ]
    assert set(comparison.reports) == {
        "baseline",
        "add_no_subsidies",
        "interim_ir",
        "low_type_likely",
    }
    assert comparison.reports["baseline"].objective_value == pytest.approx(1.5, abs=1e-9)
    assert comparison.reports["add_no_subsidies"].objective_value == pytest.approx(1.5, abs=1e-9)
    assert comparison.reports["interim_ir"].objective_value == pytest.approx(1.5, abs=1e-9)
    assert comparison.reports["low_type_likely"].objective_value == pytest.approx(1.0, abs=1e-9)

    frame = comparison.to_pandas()
    assert list(frame.index) == [
        "baseline",
        "add_no_subsidies",
        "interim_ir",
        "low_type_likely",
    ]
    assert frame.loc["low_type_likely", "audit_passed"]


def test_failed_scenario_is_labeled_and_json_serializable() -> None:
    comparison = _base_problem().compare_scenarios(
        {
            "baseline": {},
            # Ex-post IR caps the low type's payment at its value, while this
            # floor requires every report to pay at least four.
            "infeasible_floor": {"add": [PaymentBounds(lower=4.0)]},
        }
    )

    baseline, failed = comparison.results
    assert baseline.status == "optimal"
    assert baseline.solution is not None
    assert failed.status == "failed"
    assert failed.solution is None
    assert failed.error is not None
    assert "InfeasibleDesignError" in failed.error

    payload = comparison.to_dict()
    assert payload["scenarios"][1]["status"] == "failed"
    assert payload["scenarios"][1]["error"] == failed.error

    serialized = json.loads(comparison.to_json())
    assert serialized["report_kind"] == "scenario_comparison"
    assert serialized["scenarios"][1]["status"] == "failed"
