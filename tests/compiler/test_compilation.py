"""Integration tests for semantic finite-mechanism LP compilation."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from mechaudit.audit.engine import audit_mechanism
from mechaudit.core.environment import FiniteEnvironment
from mechaudit.core.valuations import PrivateValues
from mechaudit.design.compiler import CompiledDesign, compile_design_problem
from mechaudit.design.sparse_model import RowSense
from mechaudit.mechanisms.direct import DirectMechanism
from mechaudit.objectives import ExpectedRevenue
from mechaudit.requirements.base import Requirement
from mechaudit.requirements.incentive import BIC, DSIC, ExPostIC
from mechaudit.requirements.participation import ExAnteIR, ExPostIR, InterimIR


def _two_by_two_environment() -> FiniteEnvironment:
    return FiniteEnvironment(
        agents={"a": ["low", "high"], "b": ["low", "high"]},
        outcomes=["off", "on"],
        prior={
            ("low", "low"): 0.20,
            ("low", "high"): 0.30,
            ("high", "low"): 0.10,
            ("high", "high"): 0.40,
        },
        values=PrivateValues(
            lambda agent, type_value, outcome: (
                {"low": 1.0, "high": 3.0}[type_value] * ({"a": 1.0, "b": 1.5}[agent])
                if outcome == "on"
                else 0.0
            )
        ),
    )


@pytest.mark.parametrize(
    ("requirement", "family", "expected_rows", "semantic_fields"),
    [
        (BIC(), "bic", 4, {"agent", "true_type", "deviation", "conditioning_mass"}),
        (
            DSIC(),
            "dsic",
            8,
            {"agent", "true_type", "deviation", "others_report"},
        ),
        (
            ExPostIC(),
            "ex_post_ic",
            8,
            {"agent", "true_type", "deviation", "true_profile"},
        ),
        (
            ExPostIR(),
            "ex_post_ir",
            8,
            {"agent", "true_type", "true_profile"},
        ),
        (
            InterimIR(),
            "interim_ir",
            4,
            {"agent", "true_type", "conditioning_mass"},
        ),
        (ExAnteIR(), "ex_ante_ir", 2, {"agent"}),
    ],
)
def test_two_by_two_compilation_counts_and_semantic_metadata(
    requirement: Requirement,
    family: str,
    expected_rows: int,
    semantic_fields: set[str],
) -> None:
    environment = _two_by_two_environment()
    compiled = compile_design_problem(environment, ExpectedRevenue(), [requirement])

    # R(K+n) = 4(2+2) = 16 variables. Allocation normalization contributes
    # one nonclassifiable equality per report profile.
    assert compiled.layout.n_columns == 16
    assert compiled.model.n_columns == 16
    assert len(compiled.rows_for_family("allocation_normalization")) == 4
    assert len(compiled.rows_for_family(family)) == expected_rows
    assert compiled.model.n_rows == 4 + expected_rows

    normalization_rows = [
        compiled.model.rows[index] for index in compiled.rows_for_family("allocation_normalization")
    ]
    assert all(row.sense is RowSense.EQUAL for row in normalization_rows)
    assert all(not row.classifiable for row in normalization_rows)

    economic_rows = [compiled.model.rows[index] for index in compiled.rows_for_family(family)]
    assert all(row.sense is RowSense.GREATER_EQUAL for row in economic_rows)
    assert all(row.classifiable for row in economic_rows)
    assert all(semantic_fields <= set(row.metadata_dict) for row in economic_rows)
    assert len({row.key for row in economic_rows}) == expected_rows

    if family in {"bic", "interim_ir"}:
        assert all(row.scale.report_unit == "conditional_utility" for row in economic_rows)
        assert all(row.scale.report_factor > 1.0 for row in economic_rows)


def _test_mechanism(environment: FiniteEnvironment) -> DirectMechanism:
    allocation = {}
    payments = {}
    for profile in environment.report_profiles:
        high_count = sum(profile[agent] == "high" for agent in environment.agents)
        on_probability = {0: 0.15, 1: 0.55, 2: 0.85}[high_count]
        allocation[profile] = {"off": 1.0 - on_probability, "on": on_probability}
        payments[profile] = {
            "a": 0.12 + 0.21 * (profile["a"] == "high"),
            "b": -0.04 + 0.17 * (profile["b"] == "high"),
        }
    return DirectMechanism(
        allocation=allocation,
        payments=payments,
        environment=environment,
    )


def _reported_compiler_slacks(
    compiled: CompiledDesign, mechanism: DirectMechanism, families: Iterable[str]
) -> dict[str, list[float]]:
    family_set = set(families)
    vector = compiled.vector_from_mechanism(mechanism)
    residuals = compiled.model.evaluate_primal(vector)
    result = {family: [] for family in family_set}
    for index, row in enumerate(compiled.model.rows):
        if row.family in family_set:
            result[row.family].append(
                row.scale.original_slack_to_reported(float(residuals.row_slack[index]))
            )
    return {family: sorted(values) for family, values in result.items()}


def test_compiler_and_independent_audit_agree_on_economic_slacks() -> None:
    environment = _two_by_two_environment()
    requirements = (
        BIC(),
        DSIC(),
        ExPostIC(),
        ExPostIR(outside_option=0.07),
        InterimIR(outside_option=0.07),
        ExAnteIR(outside_option=0.07),
    )
    mechanism = _test_mechanism(environment)
    compiled = compile_design_problem(environment, ExpectedRevenue(), requirements)
    audit = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=requirements,
    )

    families = {
        "bic",
        "dsic",
        "ex_post_ic",
        "ex_post_ir",
        "interim_ir",
        "ex_ante_ir",
    }
    compiled_slacks = _reported_compiler_slacks(compiled, mechanism, families)
    audited_slacks = {
        family: sorted(
            float(check.slack)
            for check in audit.checks
            if check.family == family and check.slack is not None
        )
        for family in families
    }

    assert compiled_slacks.keys() == audited_slacks.keys()
    for family in families:
        assert compiled_slacks[family] == pytest.approx(audited_slacks[family], abs=1e-12)
