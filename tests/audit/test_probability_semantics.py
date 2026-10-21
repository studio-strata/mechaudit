from __future__ import annotations

import pytest

from mechaudit import (
    BIC,
    DirectMechanism,
    ExpectedSocialWelfare,
    FiniteEnvironment,
    InterimIR,
    PrivateValues,
)
from mechaudit.audit import CheckStatus


def test_small_positive_marginal_is_conditioned_not_treated_as_zero() -> None:
    rare_mass = 1e-14
    environment = FiniteEnvironment(
        agents={"agent": ("rare", "common")},
        outcomes=("outcome",),
        prior={"rare": rare_mass, "common": 1.0 - rare_mass},
        values=PrivateValues(lambda _agent, _type, _outcome: 0.0),
    )
    mechanism = DirectMechanism(
        allocation={
            "rare": {"outcome": 1.0},
            "common": {"outcome": 1.0},
        },
        payments={"rare": 0.0, "common": 0.0},
    )

    audit = environment.audit(mechanism, (BIC(), InterimIR()))

    assert audit.passed
    assert len(audit.ic) == 2
    assert len(audit.ir) == 2
    assert all(check.status is CheckStatus.PASS for check in (*audit.ic, *audit.ir))
    rare_checks = [
        check for check in (*audit.ic, *audit.ir) if check.metadata.get("true_type") == "rare"
    ]
    assert rare_checks
    assert all(
        check.metadata["conditioning_mass"] == pytest.approx(rare_mass) for check in rare_checks
    )
    assert any("probability tolerance" in warning for warning in audit.numerical_health.warnings)

    compiled = environment.design(
        objective=ExpectedSocialWelfare(),
        requirements=(BIC(), InterimIR()),
    ).compile()
    conditional_rows = [
        row
        for row in compiled.model.rows
        if row.family in {"bic", "interim_ir"}
        and dict(row.metadata).get("conditioning_mass") == rare_mass
    ]
    assert len(conditional_rows) == 2
    assert all(
        row.scale.report_factor == pytest.approx(1.0 / rare_mass) for row in conditional_rows
    )


def _belief_warnings(audit: object) -> list[str]:
    health = audit.numerical_health  # type: ignore[attr-defined]
    return [warning for warning in health.warnings if "conditional-belief" in warning]


def test_independent_priors_do_not_raise_conditional_belief_warnings() -> None:
    mechanism = DirectMechanism(
        allocation={
            profile: {"outcome": 1.0}
            for profile in (("l", "x"), ("l", "y"), ("h", "x"), ("h", "y"))
        },
        payments={
            profile: {"a": 0.0, "b": 0.0}
            for profile in (("l", "x"), ("l", "y"), ("h", "x"), ("h", "y"))
        },
    )
    one_agent = FiniteEnvironment(
        agents={"agent": ("low", "high")},
        outcomes=("outcome",),
        prior={"low": 0.5, "high": 0.5},
        values=PrivateValues(lambda _agent, _type, _outcome: 0.0),
    )
    one_agent_audit = one_agent.audit(
        DirectMechanism(
            allocation={"low": {"outcome": 1.0}, "high": {"outcome": 1.0}},
            payments={"low": 0.0, "high": 0.0},
        ),
        (BIC(),),
    )
    independent = FiniteEnvironment(
        agents={"a": ("l", "h"), "b": ("x", "y")},
        outcomes=("outcome",),
        prior={("l", "x"): 0.1, ("l", "y"): 0.4, ("h", "x"): 0.1, ("h", "y"): 0.4},
        values=PrivateValues(lambda _agent, _type, _outcome: 0.0),
    )
    correlated = independent.with_prior(
        {("l", "x"): 0.3, ("l", "y"): 0.2, ("h", "x"): 0.2, ("h", "y"): 0.3}
    )
    nearly_independent = independent.with_prior(
        {
            ("l", "x"): 0.1 + 1e-9,
            ("l", "y"): 0.4 - 1e-9,
            ("h", "x"): 0.1 - 1e-9,
            ("h", "y"): 0.4 + 1e-9,
        }
    )

    assert _belief_warnings(one_agent_audit) == []
    assert one_agent_audit.numerical_health.details["independent_prior"] is True
    assert _belief_warnings(independent.audit(mechanism, (BIC(),))) == []
    assert _belief_warnings(correlated.audit(mechanism, (BIC(),))) == []
    fragile = nearly_independent.audit(mechanism, (BIC(),))
    assert fragile.numerical_health.details["independent_prior"] is False
    assert _belief_warnings(fragile)
