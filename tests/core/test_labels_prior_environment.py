from __future__ import annotations

import math

import pytest

from mechaudit.core.environment import FiniteEnvironment
from mechaudit.core.labels import (
    ReportProfile,
    StructuredLabel,
    TypeProfile,
    freeze_label,
    thaw_label,
)
from mechaudit.core.prior import JointPrior
from mechaudit.core.valuations import InterdependentValues, PrivateValues
from mechaudit.exceptions import LabelValidationError, PriorValidationError


def _zero_private_value(agent: object, type_value: object, outcome: object) -> float:
    del agent, type_value, outcome
    return 0.0


def test_structured_labels_are_stable_hashable_and_order_independent() -> None:
    left = freeze_label({"value": 3, "attributes": ["red", {"size": 2}]})
    right = freeze_label({"attributes": ["red", {"size": 2}], "value": 3})

    assert isinstance(left, StructuredLabel)
    assert left == right
    assert hash(left) == hash(right)
    assert tuple(left) == ("attributes", "value")
    assert thaw_label(left) == {
        "attributes": ["red", {"size": 2}],
        "value": 3,
    }

    with pytest.raises(LabelValidationError, match="sets are not valid labels"):
        freeze_label({"unstable"})
    with pytest.raises(LabelValidationError, match="NaN is not a valid label"):
        freeze_label(math.nan)


def test_one_agent_mapping_type_is_not_confused_with_a_profile_mapping() -> None:
    low = {"value": 1, "demand": "low"}
    high = {"demand": "high", "value": 3}
    environment = FiniteEnvironment(
        agents={"buyer": [low, high]},
        outcomes=["no_sale", "sale"],
        values=PrivateValues(
            lambda agent, type_value, outcome: (
                float(type_value["value"]) if agent == "buyer" and outcome == "sale" else 0.0
            )
        ),
    )

    low_profile = environment.coerce_profile(low)
    low_report = environment.coerce_report_profile(low)
    assert isinstance(low_profile, TypeProfile)
    assert isinstance(low_report, ReportProfile)
    assert low_profile["buyer"] == freeze_label(low)
    assert environment.profile_index(low) == 0
    assert environment.profile_index(high) == 1
    assert environment.evaluate_value("buyer", high, "sale") == 3.0

    equivalent_environment = FiniteEnvironment(
        agents={"buyer": [{"demand": "low", "value": 1}, {"value": 3, "demand": "high"}]},
        outcomes=["no_sale", "sale"],
        values=PrivateValues(_zero_private_value),
    )
    assert environment.domain_fingerprint == equivalent_environment.domain_fingerprint


def test_independent_prior_has_canonical_product_order_and_marginals() -> None:
    prior = JointPrior.from_independent(
        {
            "buyer": {"low": 0.25, "high": 0.75},
            "seller": {"low": 0.4, "high": 0.6},
        }
    )

    expected = {
        ("low", "low"): 0.10,
        ("low", "high"): 0.15,
        ("high", "low"): 0.30,
        ("high", "high"): 0.45,
    }
    assert [profile.values_tuple for profile in prior] == list(expected)
    for profile, probability in prior.items():
        assert probability == pytest.approx(expected[profile.values_tuple])
    assert prior.marginal("buyer", "high") == pytest.approx(0.75)
    assert prior.marginal("seller", "low") == pytest.approx(0.4)
    assert prior.zero_marginal_types() == ()


def test_correlated_prior_and_raw_environment_prior_mapping() -> None:
    probabilities = {
        ("low", "a"): 0.4,
        ("low", "b"): 0.1,
        ("high", "a"): 0.1,
        ("high", "b"): 0.4,
    }
    prior = JointPrior.from_mapping(
        probabilities,
        agent_order=("buyer", "signal"),
        type_spaces={"buyer": ("low", "high"), "signal": ("a", "b")},
    )
    assert prior.marginal("buyer", "low") == pytest.approx(0.5)
    assert prior.marginal("signal", "a") == pytest.approx(0.5)
    assert prior[TypeProfile.from_values(("buyer", "signal"), ("low", "b"))] == 0.1

    environment = FiniteEnvironment(
        agents={"buyer": ("low", "high"), "signal": ("a", "b")},
        outcomes=("no_sale", "sale"),
        prior=probabilities,
        values=PrivateValues(_zero_private_value),
    )
    assert isinstance(environment.prior, JointPrior)
    assert environment.prior == prior
    assert environment.prior.marginal("buyer", "high") == pytest.approx(0.5)


def test_missing_prior_profiles_require_an_explicit_zero_fill_policy() -> None:
    probabilities = {("low", "a"): 1.0}
    arguments = {
        "agent_order": ("buyer", "signal"),
        "type_spaces": {"buyer": ("low", "high"), "signal": ("a", "b")},
    }
    with pytest.raises(PriorValidationError, match="omits Cartesian-product profiles"):
        JointPrior.from_mapping(probabilities, **arguments)

    completed = JointPrior.from_mapping(probabilities, missing="zero", **arguments)
    assert len(completed) == 4
    assert completed.zero_marginal_types() == (
        ("buyer", "high"),
        ("signal", "b"),
    )

    with pytest.raises(PriorValidationError, match="probability cannot be negative"):
        JointPrior.from_mapping(
            {("low", "a"): 1.1, ("low", "b"): -0.1},
            agent_order=("buyer", "signal"),
            missing="zero",
        )


def test_environment_eagerly_tabulates_values_and_declares_information_kind() -> None:
    multiplier = [1.0]
    environment = FiniteEnvironment(
        agents={"buyer": ("low", "high")},
        outcomes=("no_sale", "sale"),
        values=PrivateValues(
            lambda agent, type_value, outcome: (
                multiplier[0] * {"low": 1.0, "high": 2.0}[type_value]
                if agent == "buyer" and outcome == "sale"
                else 0.0
            )
        ),
    )
    multiplier[0] = 100.0

    assert environment.is_private_values
    assert environment.evaluate_value("buyer", "high", "sale") == 2.0

    interdependent = FiniteEnvironment(
        agents={"buyer": ("low", "high")},
        outcomes=("no_sale", "sale"),
        values=InterdependentValues(
            lambda agent, profile, outcome: (
                2.0 if agent == "buyer" and profile[agent] == "high" and outcome == "sale" else 0.0
            )
        ),
    )
    assert not interdependent.is_private_values


def test_prior_lookup_accepts_labeled_keys_and_shorthand() -> None:
    one_agent = JointPrior.from_mapping({"low": 0.25, "high": 0.75}, agent_order=("buyer",))
    two_agents = JointPrior.from_independent(
        {"a": {"low": 0.5, "high": 0.5}, "b": {"x": 0.2, "y": 0.8}}
    )

    assert one_agent["low"] == pytest.approx(0.25)
    assert one_agent[{"buyer": "high"}] == pytest.approx(0.75)
    assert "low" in one_agent
    assert "missing" not in one_agent
    assert two_agents[("high", "y")] == pytest.approx(0.4)
    assert two_agents[{"b": "x", "a": "low"}] == pytest.approx(0.1)
    with pytest.raises(KeyError):
        _ = two_agents[("low",)]
    with pytest.raises(KeyError):
        _ = two_agents[("low", "z")]


def test_prior_independence_and_alignment_errors() -> None:
    independent = JointPrior.from_independent(
        {"a": {"low": 0.5, "high": 0.5}, "b": {"x": 0.2, "y": 0.8}}
    )
    correlated = JointPrior.from_mapping(
        {("low", "x"): 0.4, ("low", "y"): 0.1, ("high", "x"): 0.1, ("high", "y"): 0.4},
        agent_order=("a", "b"),
    )

    assert independent.is_independent()
    assert not correlated.is_independent()
    assert JointPrior.from_mapping({"low": 0.3, "high": 0.7}, agent_order=("a",)).is_independent()
    assert independent.marginal("b", "y") == pytest.approx(0.8)
    with pytest.raises(PriorValidationError, match="every agent"):
        independent.align(agents=("a", "b"), type_spaces={"a": ("low", "high")})


def test_environment_requires_values_keyword() -> None:
    with pytest.raises(TypeError):
        FiniteEnvironment(  # type: ignore[call-arg]
            agents={"buyer": ("low",)},
            outcomes=("sale",),
            value=PrivateValues(_zero_private_value),
        )


def test_other_profile_tables_match_labeled_profiles() -> None:
    environment = FiniteEnvironment(
        agents={"a": ("l", "m", "h"), "b": ("x", "y"), "c": ("p", "q")},
        outcomes=("o",),
        values=PrivateValues(_zero_private_value),
    )

    for agent_index, agent in enumerate(environment.agents):
        stride = environment._strides[agent_index]
        entries = environment._other_profiles[agent]
        assert len(entries) == environment.n_profiles // len(environment.types[agent])
        for labels, base in entries:
            for position, type_value in enumerate(environment.types[agent]):
                profile = environment.profiles[base + position * stride]
                assert profile[agent] == type_value
                assert profile.without(agent) == labels
                assert environment._type_position(agent_index, base + position * stride) == position
