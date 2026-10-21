"""Public contracts for options, outcome helpers, and linear objectives."""

from __future__ import annotations

import math

import pytest

from mechaudit import (
    AuditOptions,
    CustomLinearObjective,
    DesignProblem,
    DirectMechanism,
    ExpectedResidualSurplus,
    ExpectedRevenue,
    ExpectedSocialWelfare,
    FiniteEnvironment,
    LinearCombination,
    LinearObjective,
    OptimalFaceLevel,
    OptimalFaceOptions,
    Outcome,
    OutcomeSpace,
    PrivateValues,
    SolveMethod,
    SolveOptions,
    TolerancePolicy,
    WeightedAgentUtilities,
    bilateral_trade_outcomes,
    multiple_item_outcomes,
    procurement_outcomes,
    public_project_outcomes,
    single_item_outcomes,
)
from mechaudit.exceptions import (
    EnvironmentValidationError,
    LabelValidationError,
    ValidationError,
)
from mechaudit.requirements.base import AllocationCoordinate, PaymentCoordinate


@pytest.mark.parametrize(
    "field",
    (
        "probability",
        "feasibility",
        "activity",
        "objective_absolute",
        "objective_relative",
        "partition_primal",
        "partition_dual",
        "complementarity",
    ),
)
@pytest.mark.parametrize("invalid", (-1.0, math.inf, math.nan, True, "small"))
def test_tolerance_fields_reject_nonfinite_negative_and_nonnumeric_values(
    field: str, invalid: object
) -> None:
    with pytest.raises(ValidationError, match=field):
        TolerancePolicy(**{field: invalid})  # type: ignore[arg-type]


def test_objective_delta_uses_distinct_absolute_and_relative_tolerances() -> None:
    policy = TolerancePolicy(objective_absolute=0.2, objective_relative=0.1)

    assert policy.objective_delta(0.0) == pytest.approx(0.2)
    assert policy.objective_delta(-3.0) == pytest.approx(0.3)
    for invalid in (math.inf, math.nan, True, "large"):
        with pytest.raises(ValidationError, match="objective_value must be finite"):
            policy.objective_delta(invalid)  # type: ignore[arg-type]


def test_workflow_options_normalize_enums_and_freeze_backend_options() -> None:
    tolerances = TolerancePolicy(feasibility=2e-7)
    audit = AuditOptions(tolerances=tolerances, max_witnesses=0, include_all_rows=False)
    solve = SolveOptions(
        tolerances=tolerances,
        method="ipm",
        crossover=False,
        presolve=False,
        time_limit=3,
        threads=2,
        random_seed=7,
        backend_options={"parallel": "off", "simplex_scale_strategy": 1},
    )
    face = OptimalFaceOptions(
        level="full",
        tolerances=tolerances,
        objective_delta=0,
        dual_ranges=True,
        constraint_families=["dsic", "ex_post_ir"],
        n_jobs=-1,
    )

    assert audit.tolerances is tolerances
    assert solve.method is SolveMethod.IPM
    assert solve.time_limit == 3.0
    assert solve.backend_options == (
        ("parallel", "off"),
        ("simplex_scale_strategy", 1),
    )
    assert face.level is OptimalFaceLevel.FULL
    assert face.constraint_families == ("dsic", "ex_post_ir")
    assert face.effective_delta(100.0) == 0.0
    assert OptimalFaceOptions(tolerances=tolerances).effective_delta(3.0) == pytest.approx(
        tolerances.objective_delta(3.0)
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (lambda: AuditOptions(max_witnesses=True), "max_witnesses"),
        (lambda: AuditOptions(max_witnesses=-1), "max_witnesses"),
        (lambda: AuditOptions(include_all_rows="yes"), "include_all_rows"),
        (lambda: AuditOptions(tolerances=object()), "tolerances"),
        (lambda: SolveOptions(method="barrier"), "method"),
        (lambda: SolveOptions(tolerances=object()), "tolerances"),
        (lambda: SolveOptions(crossover="off"), "crossover"),
        (lambda: SolveOptions(presolve=1), "presolve"),
        (lambda: SolveOptions(time_limit=0), "time_limit"),
        (lambda: SolveOptions(threads=1.5), "threads"),
        (lambda: SolveOptions(random_seed=True), "random_seed"),
        (
            lambda: SolveOptions(backend_options=(("duplicate", 1), ("duplicate", 2))),
            "duplicate",
        ),
        (lambda: SolveOptions(backend_options=(("", 1),)), "nonempty"),
        (lambda: OptimalFaceOptions(level="exhaustive"), "level"),
        (lambda: OptimalFaceOptions(tolerances=object()), "tolerances"),
        (lambda: OptimalFaceOptions(dual_ranges=1), "dual_ranges"),
        (lambda: OptimalFaceOptions(constraint_families="dsic"), "constraint_families"),
        (lambda: OptimalFaceOptions(constraint_families=("",)), "constraint_families"),
        (lambda: OptimalFaceOptions(n_jobs=1.5), "n_jobs"),
        (lambda: OptimalFaceOptions(n_jobs=0), "n_jobs"),
    ),
)
def test_workflow_options_reject_invalid_public_values(factory: object, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        factory()  # type: ignore[operator]


def test_structured_outcome_and_space_are_immutable_mapping_sequence_values() -> None:
    outcome = Outcome.from_mapping({"kind": "allocation", "assignment": {"item": "buyer"}})
    same = Outcome(kind="allocation", assignment={"item": "buyer"})
    space = OutcomeSpace((outcome, Outcome(kind="allocation", assignment={})))

    assert outcome == same
    assert hash(outcome) == hash(same)
    assert outcome["kind"] == "allocation"
    assert outcome.to_dict() == {
        "assignment": {"item": "buyer"},
        "kind": "allocation",
    }
    assert tuple(space) == space.outcomes
    assert space[0].to_dict() == outcome.to_dict()  # type: ignore[union-attr]
    assert space[:] == space.outcomes

    with pytest.raises(LabelValidationError, match="supplied more than once") as caught:
        Outcome({"kind": "old"}, kind="new")
    assert caught.value.context == {"attributes": ["kind"]}
    with pytest.raises(LabelValidationError, match="cannot be empty"):
        OutcomeSpace(())
    with pytest.raises(LabelValidationError, match="duplicate"):
        OutcomeSpace((outcome, same))


def test_small_outcome_factories_have_explicit_order_and_labels() -> None:
    single = single_item_outcomes(("buyer_b", "buyer_a"))
    assert [item.to_dict() for item in single] == [
        {"kind": "single_item", "winner": None},
        {"kind": "single_item", "winner": "buyer_b"},
        {"kind": "single_item", "winner": "buyer_a"},
    ]
    assert [
        item.to_dict() for item in single_item_outcomes(("buyer",), include_unallocated=False)
    ] == [{"kind": "single_item", "winner": "buyer"}]

    assignments = multiple_item_outcomes(("a", "b"), ("x", "y"))
    assert len(assignments) == 9
    assert assignments[0].to_dict()["allocation"] == {"x": None, "y": None}
    assert assignments[-1].to_dict()["allocation"] == {"x": "b", "y": "b"}
    allocated = multiple_item_outcomes(("a", "b"), ("x", "y"), allow_unallocated=False)
    assert len(allocated) == 4
    assert all(None not in item.to_dict()["allocation"].values() for item in allocated)

    assert [item.to_dict() for item in public_project_outcomes()] == [
        {"build": False, "kind": "public_project"},
        {"build": True, "kind": "public_project"},
    ]
    assert [item.to_dict() for item in bilateral_trade_outcomes()] == [
        {"kind": "bilateral_trade", "trade": False},
        {"kind": "bilateral_trade", "trade": True},
    ]
    assert [item.to_dict() for item in procurement_outcomes(("s2", "s1"))] == [
        {"kind": "procurement", "supplier": None},
        {"kind": "procurement", "supplier": "s2"},
        {"kind": "procurement", "supplier": "s1"},
    ]
    with pytest.raises(LabelValidationError, match="agents cannot be empty"):
        single_item_outcomes(())
    with pytest.raises(LabelValidationError, match="items cannot be empty"):
        multiple_item_outcomes(("a",), ())


def _objective_environment() -> FiniteEnvironment:
    return FiniteEnvironment(
        agents={"buyer": ("only",), "seller": ("only",)},
        outcomes=("off", "trade"),
        prior={("only", "only"): 1.0},
        values=PrivateValues(
            lambda agent, _type, outcome: (
                {"buyer": 5.0, "seller": 3.0}[agent] if outcome == "trade" else 0.0
            )
        ),
    )


def _objective_value(
    environment: FiniteEnvironment, mechanism: DirectMechanism, objective: object
) -> float:
    return (
        DesignProblem(
            environment=environment,
            objective=objective,  # type: ignore[arg-type]
        )
        .compile()
        .objective_value(mechanism)
    )


def test_objective_algebra_preserves_coefficients_constant_and_names() -> None:
    revenue = ExpectedRevenue()
    welfare = ExpectedSocialWelfare()
    combination = LinearCombination(((2, revenue), (0, welfare), (-0.5, welfare)), constant=4)

    assert revenue.name == "expected_revenue"
    assert welfare.name == "expected_social_welfare"
    assert ExpectedResidualSurplus().name == "expected_residual_surplus"
    assert LinearObjective().name == "LinearObjective"
    assert combination.name == "linear_combination"
    assert combination.terms == ((2.0, revenue), (-0.5, welfare))
    assert combination.constant == 4.0
    assert (2 * revenue).terms == ((2.0, revenue),)
    assert (revenue * -1).terms == ((-1.0, revenue),)
    assert (revenue + welfare).terms == ((1.0, revenue), (1.0, welfare))

    with pytest.raises(ValidationError, match="cannot be empty"):
        LinearCombination(((0.0, revenue),))
    with pytest.raises(ValidationError, match="LinearObjective"):
        LinearCombination(((1.0, object()),))  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="coefficient must be finite"):
        revenue.weighted(math.inf)
    with pytest.raises(TypeError):
        _ = revenue + 1  # type: ignore[operator]


def test_residual_surplus_and_weighted_utilities_are_value_minus_payment() -> None:
    environment = _objective_environment()
    mechanism = DirectMechanism(
        allocation={
            ("only", "only"): {"off": 0.0, "trade": 1.0},
        },
        payments={
            ("only", "only"): {"buyer": 2.0, "seller": -1.0},
        },
        environment=environment,
    )
    weights = {"buyer": 2.0, "seller": 0.5}

    unweighted_welfare = _objective_value(environment, mechanism, ExpectedSocialWelfare())
    unweighted_revenue = _objective_value(environment, mechanism, ExpectedRevenue())
    unweighted_residual = _objective_value(environment, mechanism, ExpectedResidualSurplus())
    weighted_welfare = _objective_value(environment, mechanism, ExpectedSocialWelfare(weights))
    weighted_revenue = _objective_value(environment, mechanism, ExpectedRevenue(weights))
    weighted_residual = _objective_value(environment, mechanism, WeightedAgentUtilities(weights))

    assert unweighted_welfare == pytest.approx(8.0)
    assert unweighted_revenue == pytest.approx(1.0)
    assert unweighted_residual == pytest.approx(7.0)
    assert unweighted_residual == pytest.approx(unweighted_welfare - unweighted_revenue)
    assert weighted_welfare == pytest.approx(11.5)
    assert weighted_revenue == pytest.approx(3.5)
    assert weighted_residual == pytest.approx(8.0)
    assert weighted_residual == pytest.approx(weighted_welfare - weighted_revenue)
    assert _objective_value(
        environment, mechanism, WeightedAgentUtilities({"buyer": 1.0, "seller": 1.0})
    ) == pytest.approx(unweighted_residual)
    assert WeightedAgentUtilities(weights).name == "weighted_agent_utilities"
    # A partial weight map gives omitted agents weight zero.
    assert _objective_value(
        environment,
        mechanism,
        WeightedAgentUtilities({"buyer": 2.0}),
    ) == pytest.approx(6.0)
    # Residual surplus is unweighted by definition; weighting belongs to
    # WeightedAgentUtilities.
    with pytest.raises(TypeError):
        ExpectedResidualSurplus(weights)  # type: ignore[call-arg]


def test_custom_and_combination_objectives_compile_as_affine_expressions() -> None:
    environment = _objective_environment()
    report = environment.report_profiles[0]
    mechanism = DirectMechanism(
        allocation=[[0.25, 0.75]],
        payments=[[2.0, -1.0]],
        environment=environment,
    )
    custom = CustomLinearObjective(
        " policy score ",
        {
            AllocationCoordinate(report, "trade"): 3.0,
            PaymentCoordinate(report, "buyer"): -2.0,
        },
        constant=1.0,
    )
    combination = LinearCombination(
        ((2.0, ExpectedRevenue()), (1.0, ExpectedSocialWelfare())),
        constant=4.0,
    )

    assert custom.name == "policy score"
    assert _objective_value(environment, mechanism, custom) == pytest.approx(-0.75)
    assert _objective_value(environment, mechanism, combination) == pytest.approx(12.0)

    with pytest.raises(ValidationError, match="agent weights are required"):
        WeightedAgentUtilities(None)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="cannot be empty"):
        WeightedAgentUtilities({})
    with pytest.raises(ValidationError, match="weight must be a real number"):
        ExpectedRevenue({"buyer": "heavy"})  # type: ignore[dict-item]
    with pytest.raises(ValidationError, match="duplicate"):
        ExpectedRevenue((("buyer", 1.0), ("buyer", 2.0)))
    with pytest.raises(ValidationError, match="name cannot be empty"):
        CustomLinearObjective(" ", {})
    with pytest.raises(EnvironmentValidationError, match="unknown agents") as caught:
        DesignProblem(
            environment=environment,
            objective=ExpectedRevenue({"ghost": 1.0}),
        ).compile()
    assert caught.value.context == {"agents": {"ghost"}}
