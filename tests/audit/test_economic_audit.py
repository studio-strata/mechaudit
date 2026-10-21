from __future__ import annotations

import json

import pandas as pd
import pytest

from mechaudit.audit.engine import audit_mechanism
from mechaudit.audit.report import CheckStatus
from mechaudit.core.environment import FiniteEnvironment
from mechaudit.core.valuations import InterdependentValues, PrivateValues
from mechaudit.exceptions import RequirementValidationError, ZeroProbabilityTypeError
from mechaudit.mechanisms.direct import DirectMechanism
from mechaudit.requirements.budget import (
    BudgetTiming,
    NoSubsidies,
    StrongBudgetBalance,
    WeakBudgetBalance,
)
from mechaudit.requirements.incentive import BIC, DSIC, ExPostIC
from mechaudit.requirements.participation import ExAnteIR, ExPostIR, InterimIR


def _one_buyer_environment(*, prior: object | None = None) -> FiniteEnvironment:
    return FiniteEnvironment(
        agents={"buyer": ("low", "high")},
        outcomes=("no_sale", "sale"),
        prior=prior,  # type: ignore[arg-type]
        values=PrivateValues(
            lambda agent, type_value, outcome: (
                {"low": 1.0, "high": 2.0}[type_value]
                if agent == "buyer" and outcome == "sale"
                else 0.0
            )
        ),
    )


def _always_sell(payments: dict[str, float]) -> DirectMechanism:
    return DirectMechanism(
        allocation={
            "low": {"no_sale": 0.0, "sale": 1.0},
            "high": {"no_sale": 0.0, "sale": 1.0},
        },
        payments=payments,
    )


def test_dsic_pass_and_exact_epsilon_failure_witness() -> None:
    environment = _one_buyer_environment()
    truthful = audit_mechanism(
        environment=environment,
        mechanism=_always_sell({"low": 1.0, "high": 1.0}),
        requirements=(DSIC(), ExPostIR()),
    )
    assert truthful.passed
    assert truthful.ic.passed
    assert truthful.ir.passed
    assert truthful.ic.minimum_slack == pytest.approx(0.0)
    assert truthful.epsilon_ic == 0.0

    own_report_prices = _always_sell({"low": 1.0, "high": 2.0})
    failing = audit_mechanism(
        environment=environment,
        mechanism=own_report_prices,
        requirements=(DSIC(), ExPostIR()),
    )
    assert not failing.passed
    assert not failing.ic.passed
    assert failing.ir.passed
    assert failing.ic.minimum_slack == pytest.approx(-1.0)
    assert failing.epsilon_ic == pytest.approx(1.0)

    witness = failing.ic.worst_deviations(k=1).iloc[0]
    assert witness["agent"] == "buyer"
    assert witness["true_type"] == "high"
    assert witness["deviation"] == "low"
    assert witness["truthful_utility"] == pytest.approx(0.0)
    assert witness["deviation_utility"] == pytest.approx(1.0)
    assert witness["deviation_gain"] == pytest.approx(1.0)


def _correlated_environment_and_mechanism() -> tuple[FiniteEnvironment, DirectMechanism]:
    probabilities = {
        ("low", "a"): 0.4,
        ("low", "b"): 0.1,
        ("high", "a"): 0.1,
        ("high", "b"): 0.4,
    }
    environment = FiniteEnvironment(
        agents={"buyer": ("low", "high"), "signal": ("a", "b")},
        outcomes=("no_sale", "sale"),
        prior=probabilities,
        values=PrivateValues(
            lambda agent, type_value, outcome: (
                {"low": 1.0, "high": 2.0}[type_value]
                if agent == "buyer" and outcome == "sale"
                else 0.0
            )
        ),
    )
    buyer_payments = {
        ("low", "a"): 0.0,
        ("low", "b"): 2.0,
        ("high", "a"): 2.0,
        ("high", "b"): 0.0,
    }
    mechanism = DirectMechanism(
        allocation={profile: {"no_sale": 0.0, "sale": 1.0} for profile in buyer_payments},
        payments={
            profile: {"buyer": payment, "signal": 0.0}
            for profile, payment in buyer_payments.items()
        },
    )
    return environment, mechanism


def test_correlated_bic_reports_conditional_and_joint_slack() -> None:
    environment, mechanism = _correlated_environment_and_mechanism()
    report = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(BIC(agents=("buyer",)),),
    )
    assert report.passed
    assert len(report.ic) == 2
    assert report.ic.minimum_slack == pytest.approx(1.2)
    for check in report.ic:
        assert check.unit == "conditional_utility"
        assert check.slack == pytest.approx(1.2)
        assert check.metadata["joint_weighted_slack"] == pytest.approx(0.6)
        assert check.metadata["conditioning_mass"] == pytest.approx(0.5)

    dsic = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(DSIC(agents=("buyer",)),),
    )
    assert not dsic.ic.passed
    assert dsic.ic.minimum_slack == pytest.approx(-2.0)
    assert dsic.epsilon_ic == pytest.approx(2.0)


def test_ex_post_interim_and_ex_ante_ir_have_distinct_timing() -> None:
    environment, mechanism = _correlated_environment_and_mechanism()
    report = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(
            ExPostIR(agents=("buyer",)),
            InterimIR(agents=("buyer",)),
            ExAnteIR(agents=("buyer",)),
        ),
    )

    ex_post = report.section("ex_post", {"ex_post_ir"})
    interim = report.section("interim", {"interim_ir"})
    ex_ante = report.section("ex_ante", {"ex_ante_ir"})
    assert not ex_post.passed
    assert ex_post.minimum_slack == pytest.approx(-1.0)
    assert interim.passed
    assert sorted(check.slack for check in interim if check.slack is not None) == pytest.approx(
        [0.6, 1.6]
    )
    assert ex_ante.passed
    assert ex_ante.minimum_slack == pytest.approx(1.1)


def test_budget_sign_and_ex_post_versus_ex_ante_timing() -> None:
    environment = _one_buyer_environment(prior={"low": 0.5, "high": 0.5})
    mechanism = DirectMechanism(
        allocation={
            "low": {"no_sale": 1.0, "sale": 0.0},
            "high": {"no_sale": 1.0, "sale": 0.0},
        },
        payments={"low": -1.0, "high": 1.0},
    )

    ex_post = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(
            NoSubsidies(),
            WeakBudgetBalance(timing=BudgetTiming.EX_POST),
            StrongBudgetBalance(timing=BudgetTiming.EX_POST),
        ),
    )
    no_subsidies = ex_post.section("no_subsidies", {"no_subsidies"})
    weak = ex_post.section("weak", {"weak_budget_balance"})
    strong = ex_post.section("strong", {"strong_budget_balance"})
    assert no_subsidies.minimum_slack == pytest.approx(-1.0)
    assert not no_subsidies.passed
    assert sorted(check.slack for check in weak if check.slack is not None) == [-1.0, 1.0]
    assert sorted(check.residual for check in strong if check.residual is not None) == [-1.0, 1.0]

    ex_ante = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(
            WeakBudgetBalance(timing=BudgetTiming.EX_ANTE),
            StrongBudgetBalance(timing=BudgetTiming.EX_ANTE),
        ),
    )
    assert ex_ante.budget.passed
    assert ex_ante.section("weak", {"weak_budget_balance"}).minimum_slack == 0.0
    assert ex_ante.section("strong", {"strong_budget_balance"}).maximum_residual == 0.0
    for check in ex_ante.budget:
        assert check.metadata["designer_revenue"] == pytest.approx(0.0)


def _zero_probability_environment_and_mechanism() -> tuple[FiniteEnvironment, DirectMechanism]:
    environment = FiniteEnvironment(
        agents={"buyer": ("ghost", "high")},
        outcomes=("no_sale", "sale"),
        prior={"ghost": 0.0, "high": 1.0},
        values=PrivateValues(
            lambda agent, type_value, outcome: (
                {"ghost": 1.0, "high": 2.0}[type_value]
                if agent == "buyer" and outcome == "sale"
                else 0.0
            )
        ),
    )
    mechanism = DirectMechanism(
        allocation={
            "ghost": {"no_sale": 1.0, "sale": 0.0},
            "high": {"no_sale": 0.0, "sale": 1.0},
        },
        payments={"ghost": 0.0, "high": 0.0},
    )
    return environment, mechanism


def test_zero_probability_bic_policies_are_explicit() -> None:
    environment, mechanism = _zero_probability_environment_and_mechanism()
    with pytest.raises(ZeroProbabilityTypeError, match="zero-marginal type"):
        audit_mechanism(
            environment=environment,
            mechanism=mechanism,
            requirements=(BIC(),),
        )

    vacuous = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(BIC(zero_probability_types="vacuous"),),
    )
    assert vacuous.ic.passed
    assert len([check for check in vacuous.ic if check.status is CheckStatus.VACUOUS]) == 1

    enforced = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(BIC(zero_probability_types="enforce_ex_post"),),
    )
    assert not enforced.ic.passed
    assert enforced.ic.minimum_slack == pytest.approx(-1.0)
    assert enforced.epsilon_ic == pytest.approx(1.0)
    failing = min(
        (check for check in enforced.ic if check.slack is not None),
        key=lambda check: check.slack or 0.0,
    )
    assert failing.metadata["zero_probability_policy"] == "enforce_ex_post"


def test_zero_probability_interim_ir_policies_are_explicit() -> None:
    environment, mechanism = _zero_probability_environment_and_mechanism()
    with pytest.raises(ZeroProbabilityTypeError, match="zero-marginal type"):
        audit_mechanism(
            environment=environment,
            mechanism=mechanism,
            requirements=(InterimIR(outside_option=0.5),),
        )

    vacuous = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(InterimIR(outside_option=0.5, zero_probability_types="vacuous"),),
    )
    assert vacuous.ir.passed
    assert len([check for check in vacuous.ir if check.status is CheckStatus.VACUOUS]) == 1

    enforced = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(InterimIR(outside_option=0.5, zero_probability_types="enforce_ex_post"),),
    )
    assert not enforced.ir.passed
    assert enforced.ir.minimum_slack == pytest.approx(-0.5)


def test_interdependent_environment_rejects_dsic_but_supports_ex_post_ic() -> None:
    environment = FiniteEnvironment(
        agents={"buyer": ("low", "high")},
        outcomes=("no_sale", "sale"),
        values=InterdependentValues(
            lambda agent, profile, outcome: (
                {"low": 1.0, "high": 2.0}[profile[agent]]
                if agent == "buyer" and outcome == "sale"
                else 0.0
            )
        ),
    )
    mechanism = _always_sell({"low": 1.0, "high": 1.0})

    with pytest.raises(RequirementValidationError, match="use ExPostIC"):
        audit_mechanism(
            environment=environment,
            mechanism=mechanism,
            requirements=(DSIC(),),
        )

    ex_post = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(ExPostIC(),),
    )
    assert ex_post.ic.passed
    assert len(ex_post.ic) == 2
    assert ex_post.ic.minimum_slack == 0.0


def test_invalid_allocation_is_reported_by_baseline_feasibility_checks() -> None:
    environment = _one_buyer_environment()
    mechanism = DirectMechanism(
        allocation={
            "low": {"no_sale": -0.25, "sale": 1.25},
            "high": {"no_sale": 0.0, "sale": 0.75},
        },
        payments={"low": 0.0, "high": 0.0},
    )
    report = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(),
    )
    assert not report.passed
    assert report.feasibility.minimum_slack == pytest.approx(-0.25)
    assert report.feasibility.maximum_residual == pytest.approx(0.25)
    assert any(
        check.family == "allocation_nonnegative" and check.status is CheckStatus.FAIL
        for check in report.checks
    )
    assert any(
        check.family == "allocation_normalization" and check.status is CheckStatus.FAIL
        for check in report.checks
    )


def test_report_dataframe_json_and_fingerprint_contracts() -> None:
    environment = _one_buyer_environment()
    mechanism = _always_sell({"low": 1.0, "high": 2.0}).bind(environment)
    report = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(DSIC(), ExPostIR()),
    )

    frames = report.to_dataframes()
    assert set(frames) == {"checks", "ic", "ir", "budget", "feasibility"}
    assert all(isinstance(frame, pd.DataFrame) for frame in frames.values())
    assert {"constraint_id", "family", "status", "tolerance"}.issubset(frames["checks"].columns)
    assert {"agent", "true_type", "deviation", "deviation_gain"}.issubset(frames["ic"].columns)

    decoded = json.loads(report.to_json())
    assert decoded["schema_version"] == "1.0"
    assert decoded["report_kind"] == "mechanism_audit"
    assert decoded["environment_fingerprint"] == environment.domain_fingerprint
    assert decoded["mechanism_fingerprint"] == mechanism.fingerprint
    assert decoded["payment_sign_convention"] == "positive_to_designer"
    assert decoded["epsilon_ic"] == pytest.approx(1.0)
    assert decoded["checks"]
    assert report.to_json() == report.to_json()
    assert "Smallest valid epsilon:" in report.summary()
