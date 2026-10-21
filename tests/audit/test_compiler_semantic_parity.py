"""Adversarial formula-level parity tests for compilation and direct auditing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import pytest

from mechaudit.audit.engine import audit_mechanism
from mechaudit.core.environment import FiniteEnvironment
from mechaudit.core.labels import AgentId, TypeProfile
from mechaudit.core.valuations import InterdependentValues, PrivateValues
from mechaudit.design.compiler import compile_design_problem
from mechaudit.exceptions import RequirementValidationError
from mechaudit.mechanisms.direct import DirectMechanism
from mechaudit.objectives import ExpectedRevenue
from mechaudit.requirements.base import AllocationCoordinate, PaymentCoordinate
from mechaudit.requirements.budget import BudgetTiming, StrongBudgetBalance, WeakBudgetBalance
from mechaudit.requirements.custom import CustomLinearConstraint, CustomLinearRequirement
from mechaudit.requirements.feasibility import Capacity, Symmetry, SymmetryOperation
from mechaudit.requirements.incentive import BIC, DSIC, ExPostIC
from mechaudit.requirements.participation import ExAnteIR, ExPostIR, InterimIR


def _assert_semantic_parity(
    environment: FiniteEnvironment,
    mechanism: DirectMechanism,
    requirements: Iterable[object],
    families: set[str],
) -> dict[tuple[str, str], list[float]]:
    """Match every compiled row to the audit check with the same identifier.

    Comparing constraint by constraint, rather than sorted value lists, also
    catches a value attached to the wrong economic constraint.
    """

    declared = tuple(requirements)
    compiled = compile_design_problem(environment, ExpectedRevenue(), declared)
    vector = compiled.vector_from_mechanism(mechanism)
    residuals = compiled.model.evaluate_primal(vector)
    compiled_rows: dict[str, tuple[str, str, float]] = {}
    for index, row in enumerate(compiled.model.rows):
        if row.family in families:
            compiled_rows[row.key] = (
                row.family,
                row.sense.value,
                row.scale.original_slack_to_reported(float(residuals.row_slack[index])),
            )

    audit = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=declared,
    )
    audited_rows: dict[str, tuple[str, str, float]] = {}
    for check in audit.checks:
        if check.family not in families:
            continue
        value = check.slack if check.slack is not None else check.residual
        assert value is not None
        audited_rows[check.constraint_id] = (check.family, check.relation, float(value))

    assert compiled_rows.keys() == audited_rows.keys()
    result: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for constraint_id, (family, relation, compiled_value) in compiled_rows.items():
        audited_family, audited_relation, audited_value = audited_rows[constraint_id]
        assert (family, relation) == (audited_family, audited_relation), constraint_id
        assert compiled_value == pytest.approx(audited_value, abs=1e-12), constraint_id
        result[(family, relation)].append(audited_value)
    return {key: sorted(values) for key, values in result.items()}


def test_multi_agent_dsic_varies_other_reports() -> None:
    environment = FiniteEnvironment(
        agents={"a": ("low", "high"), "b": ("x", "y")},
        outcomes=("outcome",),
        prior={
            profile: 0.25 for profile in (("low", "x"), ("low", "y"), ("high", "x"), ("high", "y"))
        },
        values=PrivateValues(lambda _agent, _type, _outcome: 0.0),
    )
    payments = {
        ("low", "x"): {"a": 0.0, "b": 0.0},
        ("high", "x"): {"a": 1.0, "b": 0.0},
        ("low", "y"): {"a": 3.0, "b": 0.0},
        ("high", "y"): {"a": 0.0, "b": 0.0},
    }
    mechanism = DirectMechanism(
        allocation={profile: {"outcome": 1.0} for profile in payments},
        payments=payments,
        environment=environment,
    )

    values = _assert_semantic_parity(environment, mechanism, (DSIC(agents=("a",)),), {"dsic"})
    assert values[("dsic", ">=")] == pytest.approx([-3.0, -1.0, 1.0, 3.0])


def test_interdependent_ex_post_ic_uses_true_full_profile_and_truthful_others() -> None:
    environment = FiniteEnvironment(
        agents={"a": ("low", "high"), "b": ("x", "y")},
        outcomes=("off", "on"),
        prior={
            profile: 0.25 for profile in (("low", "x"), ("low", "y"), ("high", "x"), ("high", "y"))
        },
        values=InterdependentValues(
            lambda agent, profile, outcome: (
                {"low": 2.0, "high": 7.0}[profile["a"]] + {"x": 0.0, "y": 10.0}[profile["b"]]
                if agent == "a" and outcome == "on"
                else 0.0
            )
        ),
    )
    on_probability = {
        ("low", "x"): 0.10,
        ("high", "x"): 0.60,
        ("low", "y"): 0.25,
        ("high", "y"): 0.75,
    }
    a_payment = {
        ("low", "x"): 0.0,
        ("high", "x"): 0.4,
        ("low", "y"): 1.0,
        ("high", "y"): -0.5,
    }
    mechanism = DirectMechanism(
        allocation={
            profile: {"off": 1.0 - probability, "on": probability}
            for profile, probability in on_probability.items()
        },
        payments={profile: {"a": payment, "b": 0.0} for profile, payment in a_payment.items()},
        environment=environment,
    )

    values = _assert_semantic_parity(
        environment,
        mechanism,
        (ExPostIC(agents=("a",)),),
        {"ex_post_ic"},
    )
    assert values[("ex_post_ic", ">=")] == pytest.approx([-7.5, -0.6, 3.1, 10.0])


def _correlated_environment_and_mechanism() -> tuple[FiniteEnvironment, DirectMechanism]:
    environment = FiniteEnvironment(
        agents={"a": ("low", "high"), "b": ("x", "y")},
        outcomes=("outcome",),
        prior={
            ("low", "x"): 0.45,
            ("low", "y"): 0.05,
            ("high", "x"): 0.10,
            ("high", "y"): 0.40,
        },
        values=PrivateValues(lambda _agent, _type, _outcome: 0.0),
    )
    payments = {
        ("low", "x"): {"a": 0.0, "b": 1.0},
        ("low", "y"): {"a": 0.0, "b": -1.0},
        ("high", "x"): {"a": 2.0, "b": 3.0},
        ("high", "y"): {"a": -8.0, "b": -3.0},
    }
    mechanism = DirectMechanism(
        allocation={profile: {"outcome": 1.0} for profile in payments},
        payments=payments,
        environment=environment,
    )
    return environment, mechanism


def test_correlated_bic_and_profile_dependent_ir_outside_options() -> None:
    environment, mechanism = _correlated_environment_and_mechanism()

    def outside(agent: AgentId, profile: TypeProfile) -> float:
        if agent != "a":
            return 0.0
        return {"low": 0.2, "high": 0.4}[profile["a"]] + (1.0 if profile["b"] == "y" else 0.0)

    requirements = (
        BIC(agents=("a",)),
        ExPostIR(outside, agents=("a",)),
        InterimIR(outside, agents=("a",)),
        ExAnteIR(outside, agents=("a",)),
    )
    values = _assert_semantic_parity(
        environment,
        mechanism,
        requirements,
        {"bic", "ex_post_ir", "interim_ir", "ex_ante_ir"},
    )

    assert values[("bic", ">=")] == pytest.approx([1.0, 6.0])
    assert values[("ex_post_ir", ">=")] == pytest.approx([-2.4, -1.2, -0.2, 6.6])
    assert values[("interim_ir", ">=")] == pytest.approx([-0.3, 4.8])
    assert values[("ex_ante_ir", ">=")] == pytest.approx([2.25])


def test_ex_ante_budget_capacity_symmetry_and_custom_signs() -> None:
    environment = FiniteEnvironment(
        agents={"a": ("low", "high"), "b": ("low", "high")},
        outcomes=("none", "a_wins", "b_wins"),
        prior={
            ("low", "low"): 0.45,
            ("low", "high"): 0.05,
            ("high", "low"): 0.10,
            ("high", "high"): 0.40,
        },
        values=PrivateValues(lambda _agent, _type, _outcome: 0.0),
    )
    allocations = {
        ("low", "low"): {"none": 0.2, "a_wins": 0.5, "b_wins": 0.3},
        ("low", "high"): {"none": 0.1, "a_wins": 0.7, "b_wins": 0.2},
        ("high", "low"): {"none": 0.4, "a_wins": 0.1, "b_wins": 0.5},
        ("high", "high"): {"none": 0.3, "a_wins": 0.6, "b_wins": 0.1},
    }
    payments = {
        ("low", "low"): {"a": 1.0, "b": 2.0},
        ("low", "high"): {"a": 3.0, "b": 4.0},
        ("high", "low"): {"a": 5.0, "b": 6.0},
        ("high", "high"): {"a": 7.0, "b": 8.0},
    }
    mechanism = DirectMechanism(
        allocation=allocations,
        payments=payments,
        environment=environment,
    )
    selected = environment.coerce_report_profile(("low", "high"))
    allocation = AllocationCoordinate(selected, "a_wins")
    payment_a = PaymentCoordinate(selected, "a")
    payment_b = PaymentCoordinate(selected, "b")
    upper = CustomLinearConstraint(
        "policy",
        {allocation: 2.0, payment_a: -1.0},
        sense="<=",
        rhs=0.5,
        constant=0.3,
        metadata={"side": "upper"},
    )
    lower = CustomLinearConstraint(
        "policy",
        {allocation: 2.0, payment_a: -1.0},
        sense=">=",
        rhs=-2.0,
        constant=0.3,
        metadata={"side": "lower"},
    )
    accounting = CustomLinearConstraint(
        "accounting",
        {payment_b: 1.0},
        sense="==",
        rhs=4.0,
    )
    custom = CustomLinearRequirement("policy", (upper, lower, accounting))
    swap = SymmetryOperation(
        name="swap",
        agents={"a": "b", "b": "a"},
        types={
            "a": {"low": "low", "high": "high"},
            "b": {"low": "low", "high": "high"},
        },
        outcomes={"none": "none", "a_wins": "b_wins", "b_wins": "a_wins"},
    )
    requirements = (
        WeakBudgetBalance(timing=BudgetTiming.EX_ANTE, floor=8.6),
        StrongBudgetBalance(timing=BudgetTiming.EX_ANTE, target=8.9),
        Capacity(
            resource="capacity",
            usage={"none": 0.0, "a_wins": 2.0, "b_wins": -1.0},
            upper=0.6,
            lower=-0.2,
            profiles=(selected,),
        ),
        Symmetry(swap),
        custom,
    )
    values = _assert_semantic_parity(
        environment,
        mechanism,
        requirements,
        {
            "weak_budget_balance",
            "strong_budget_balance",
            "capacity",
            "symmetry",
            "custom",
        },
    )

    # Expected total payment is 8.8 under the correlated prior.
    assert values[("weak_budget_balance", ">=")] == pytest.approx([0.2])
    assert values[("strong_budget_balance", "==")] == pytest.approx([-0.1])
    assert values[("capacity", ">=")] == pytest.approx([-0.6, 1.4])
    # Same display name, different stable metadata: both custom rows survive.
    assert values[("custom", ">=")] == pytest.approx([0.7, 1.8])
    assert values[("custom", "==")] == pytest.approx([0.0])
    assert len(values[("symmetry", "==")]) == 20
    assert max(abs(value) for value in values[("symmetry", "==")]) > 0.0


def test_symmetry_type_actions_must_be_bijections() -> None:
    environment = FiniteEnvironment(
        agents={"a": ("low", "high"), "b": ("low", "high")},
        outcomes=("outcome",),
        prior={
            profile: 0.25
            for profile in (
                ("low", "low"),
                ("low", "high"),
                ("high", "low"),
                ("high", "high"),
            )
        },
        values=PrivateValues(lambda _agent, _type, _outcome: 0.0),
    )
    mechanism = DirectMechanism(
        allocation={profile: {"outcome": 1.0} for profile in environment.report_profiles},
        payments={profile: {"a": 0.0, "b": 0.0} for profile in environment.report_profiles},
        environment=environment,
    )
    many_to_one = SymmetryOperation(
        name="invalid_swap",
        agents={"a": "b", "b": "a"},
        types={
            "a": {"low": "low", "high": "low"},
            "b": {"low": "low", "high": "high"},
        },
        outcomes={"outcome": "outcome"},
    )
    requirement = Symmetry(many_to_one)

    with pytest.raises(RequirementValidationError, match="bijection"):
        compile_design_problem(environment, ExpectedRevenue(), (requirement,))
    with pytest.raises(RequirementValidationError, match="bijection"):
        audit_mechanism(
            environment=environment,
            mechanism=mechanism,
            requirements=(requirement,),
        )
