from __future__ import annotations

import math

import numpy as np
import pytest

from mechaudit.core.environment import FiniteEnvironment
from mechaudit.core.valuations import InterdependentValues
from mechaudit.diagnostics.correlation import (
    correlation_diagnostics,
    cremer_mclean_diagnostic,
)
from mechaudit.exceptions import EnvironmentValidationError
from mechaudit.mechanisms.direct import DirectMechanism


def _correlated_environment(epsilon: float, *, value_scale: float = 0.0) -> FiniteEnvironment:
    return FiniteEnvironment(
        agents={"buyer": ("low", "high"), "signal": ("a", "b")},
        outcomes=("outside",),
        prior={
            ("low", "a"): 0.5 * (0.5 + epsilon),
            ("low", "b"): 0.5 * (0.5 - epsilon),
            ("high", "a"): 0.5 * (0.5 - epsilon),
            ("high", "b"): 0.5 * (0.5 + epsilon),
        },
        values=InterdependentValues(
            lambda agent, profile, outcome: (
                value_scale if agent == "buyer" and outcome == "outside" else 0.0
            )
        ),
    )


@pytest.mark.parametrize(
    ("epsilon", "expected_rank", "expected_smallest", "expected_condition"),
    [
        (0.1, 2, 0.2, 5.0),
        (1e-6, 2, 2e-6, 5e5),
        (0.0, 1, 0.0, math.inf),
    ],
)
def test_conditional_belief_rank_tracks_distance_from_independence(
    epsilon: float,
    expected_rank: int,
    expected_smallest: float,
    expected_condition: float,
) -> None:
    report = cremer_mclean_diagnostic(_correlated_environment(epsilon))
    buyer = report.for_agent("buyer")

    np.testing.assert_allclose(
        buyer.matrix,
        [[0.5 + epsilon, 0.5 - epsilon], [0.5 - epsilon, 0.5 + epsilon]],
        atol=1e-15,
    )
    assert buyer.singular_values == pytest.approx((1.0, expected_smallest), abs=1e-15)
    assert buyer.rank == expected_rank
    assert buyer.full_row_rank is (expected_rank == 2)
    assert buyer.smallest_singular_value == pytest.approx(expected_smallest, abs=1e-15)
    assert buyer.condition_number == pytest.approx(expected_condition)
    assert report.full_rank is (expected_rank == 2)

    if epsilon == 0.0:
        assert buyer.nearly_duplicate_pairs == (("low", "high", 0.0),)
        assert any("not full row rank" in warning for warning in buyer.warnings)
        assert any("nearly identical" in warning for warning in buyer.warnings)
    else:
        assert buyer.nearly_duplicate_pairs == ()


def test_rank_tolerance_controls_the_numerical_cremer_mclean_boundary() -> None:
    environment = _correlated_environment(1e-11)

    default = cremer_mclean_diagnostic(environment, rank_tolerance=1e-10).for_agent("buyer")
    strict = cremer_mclean_diagnostic(environment, rank_tolerance=1e-13).for_agent("buyer")

    assert default.rank == 1
    assert not default.full_row_rank
    assert math.isinf(default.condition_number)
    assert strict.rank == 2
    assert strict.full_row_rank
    assert strict.smallest_singular_value == pytest.approx(2e-11, rel=1e-5)
    assert strict.condition_number == pytest.approx(5e10, rel=1e-5)


def test_zero_marginal_types_are_explicitly_diagnosed() -> None:
    environment = FiniteEnvironment(
        agents={"buyer": ("ghost", "real"), "signal": ("a", "b")},
        outcomes=("outside",),
        prior={
            ("ghost", "a"): 0.0,
            ("ghost", "b"): 0.0,
            ("real", "a"): 0.5,
            ("real", "b"): 0.5,
        },
        values=InterdependentValues(lambda agent, profile, outcome: 0.0),
    )

    buyer = cremer_mclean_diagnostic(environment).for_agent("buyer")
    assert buyer.zero_marginal_types == ("ghost",)
    assert buyer.matrix[0] == pytest.approx([0.0, 0.0])
    assert buyer.rank == 1
    assert not buyer.full_row_rank
    assert any("zero-marginal types" in warning for warning in buyer.warnings)


def test_report_exports_and_payment_scale_are_stable() -> None:
    environment = _correlated_environment(0.1, value_scale=2.0)
    mechanism = DirectMechanism(
        allocation={profile.values_tuple: {"outside": 1.0} for profile in environment.profiles},
        payments={
            profile.values_tuple: {
                "buyer": 4.0 if profile["buyer"] == "high" else -1.0,
                "signal": 0.0,
            }
            for profile in environment.profiles
        },
    )

    report = cremer_mclean_diagnostic(environment, mechanism)
    buyer = report.for_agent("buyer")
    exported = report.to_dict()
    frame = buyer.to_pandas()

    assert report.maximum_absolute_payment == 4.0
    assert report.payment_to_value_ratio == 2.0
    assert exported["full_rank"] is True
    assert exported["maximum_absolute_payment"] == 4.0
    np.testing.assert_allclose(
        exported["agents"][0]["conditional_belief_matrix"],
        buyer.matrix,
    )
    assert frame.shape == (2, 2)
    np.testing.assert_allclose(frame.to_numpy(), buyer.matrix)
    assert not buyer.matrix.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        buyer.matrix[0, 0] = 0.0


def test_correlation_wrapper_and_validation_errors() -> None:
    environment = _correlated_environment(0.1)
    diagnostics = correlation_diagnostics(environment)
    assert tuple(item.agent for item in diagnostics) == environment.agents

    no_prior = FiniteEnvironment(
        agents={"buyer": ("low", "high")},
        outcomes=("outside",),
        values=InterdependentValues(lambda agent, profile, outcome: 0.0),
    )
    with pytest.raises(EnvironmentValidationError, match="require a joint prior"):
        cremer_mclean_diagnostic(no_prior)
    with pytest.raises(ValueError, match="rank tolerance must be positive"):
        cremer_mclean_diagnostic(environment, rank_tolerance=0.0)
    with pytest.raises(ValueError, match="duplicate tolerance nonnegative"):
        cremer_mclean_diagnostic(environment, duplicate_tolerance=-1.0)


def test_positive_payments_over_zero_values_report_infinite_scale_ratio() -> None:
    environment = _correlated_environment(0.1)
    mechanism = DirectMechanism.from_arrays(
        np.ones((environment.n_profiles, 1)),
        np.ones((environment.n_profiles, environment.n_agents)),
        environment=environment,
    )
    report = cremer_mclean_diagnostic(environment, mechanism)
    assert report.maximum_absolute_payment == 1.0
    assert math.isinf(report.payment_to_value_ratio)
