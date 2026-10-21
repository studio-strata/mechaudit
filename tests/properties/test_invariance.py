from __future__ import annotations

import math
from typing import NamedTuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mechaudit.audit.engine import audit_mechanism
from mechaudit.audit.report import MechanismAuditReport
from mechaudit.core.environment import FiniteEnvironment
from mechaudit.core.labels import freeze_label, normalize_axis
from mechaudit.core.prior import JointPrior
from mechaudit.core.valuations import PrivateValues
from mechaudit.exceptions import LabelValidationError, PriorValidationError
from mechaudit.mechanisms.direct import DirectMechanism
from mechaudit.requirements.incentive import DSIC
from mechaudit.requirements.participation import ExPostIR

PROPERTY_SETTINGS = settings(
    max_examples=25,
    deadline=None,
    derandomize=True,
    database=None,
)


class Menu(NamedTuple):
    values: list[int]
    allocation_quarters: list[int]
    payment_halves: list[int]
    prior_weights: list[int]


@st.composite
def finite_menus(draw: st.DrawFn) -> Menu:
    """Generate small exact-arithmetic one-agent menus."""

    size = draw(st.integers(min_value=2, max_value=4))
    return Menu(
        values=draw(st.lists(st.integers(min_value=-2, max_value=5), min_size=size, max_size=size)),
        allocation_quarters=draw(
            st.lists(st.integers(min_value=0, max_value=4), min_size=size, max_size=size)
        ),
        payment_halves=draw(
            st.lists(st.integers(min_value=-4, max_value=8), min_size=size, max_size=size)
        ),
        prior_weights=draw(
            st.lists(st.integers(min_value=1, max_value=10), min_size=size, max_size=size)
        ),
    )


def _audit_signature(
    report: MechanismAuditReport,
) -> list[tuple[str, str, float | None, float | None, str]]:
    return sorted(
        [
            (
                check.family,
                check.relation,
                check.slack,
                check.residual,
                check.status.value,
            )
            for check in report.checks
        ],
        key=lambda item: (
            item[0],
            item[1],
            math.inf if item[2] is None else item[2],
            math.inf if item[3] is None else item[3],
            item[4],
        ),
    )


@PROPERTY_SETTINGS
@given(finite_menus())
def test_audit_is_invariant_to_deterministic_domain_relabeling(menu: Menu) -> None:
    """Labels and axis order must not change an economically identical audit."""

    size = len(menu.values)
    original_types = tuple(f"t{index}" for index in range(size))
    total_weight = sum(menu.prior_weights)
    original_prior = {
        type_value: menu.prior_weights[index] / total_weight
        for index, type_value in enumerate(original_types)
    }
    original_values = dict(zip(original_types, menu.values, strict=True))
    original_environment = FiniteEnvironment(
        agents={"agent": original_types},
        outcomes=("off", "on"),
        prior=original_prior,
        values=PrivateValues(
            lambda agent, type_value, outcome: (
                float(original_values[type_value]) if agent == "agent" and outcome == "on" else 0.0
            )
        ),
    )
    original_mechanism = DirectMechanism(
        allocation={
            type_value: {
                "off": 1.0 - menu.allocation_quarters[index] / 4.0,
                "on": menu.allocation_quarters[index] / 4.0,
            }
            for index, type_value in enumerate(original_types)
        },
        payments={
            type_value: menu.payment_halves[index] / 2.0
            for index, type_value in enumerate(original_types)
        },
    )

    relabel = {
        type_value: f"renamed_type_{index}" for index, type_value in enumerate(original_types)
    }
    relabeled_types = tuple(relabel[type_value] for type_value in reversed(original_types))
    relabeled_values = {
        relabel[type_value]: original_values[type_value] for type_value in original_types
    }
    relabeled_prior = {
        relabel[type_value]: original_prior[type_value] for type_value in original_types
    }
    relabeled_environment = FiniteEnvironment(
        agents={"renamed_agent": relabeled_types},
        # Reverse the outcome axis while also changing every label.
        outcomes=("allocated", "withheld"),
        prior=relabeled_prior,
        values=PrivateValues(
            lambda agent, type_value, outcome: (
                float(relabeled_values[type_value])
                if agent == "renamed_agent" and outcome == "allocated"
                else 0.0
            )
        ),
    )
    relabeled_mechanism = DirectMechanism(
        allocation={
            relabel[type_value]: {
                "allocated": menu.allocation_quarters[index] / 4.0,
                "withheld": 1.0 - menu.allocation_quarters[index] / 4.0,
            }
            for index, type_value in enumerate(original_types)
        },
        payments={
            relabel[type_value]: menu.payment_halves[index] / 2.0
            for index, type_value in enumerate(original_types)
        },
    )

    requirements = (DSIC(), ExPostIR())
    original_report = audit_mechanism(
        environment=original_environment,
        mechanism=original_mechanism,
        requirements=requirements,
    )
    relabeled_report = audit_mechanism(
        environment=relabeled_environment,
        mechanism=relabeled_mechanism,
        requirements=requirements,
    )

    assert _audit_signature(original_report) == _audit_signature(relabeled_report)
    assert original_report.passed is relabeled_report.passed
    assert original_report.epsilon_ic == pytest.approx(relabeled_report.epsilon_ic)


@PROPERTY_SETTINGS
@given(st.sets(st.integers(), max_size=5))
def test_unordered_labels_are_always_rejected(label: set[int]) -> None:
    with pytest.raises(LabelValidationError, match="sets are not valid labels"):
        freeze_label(label)


@PROPERTY_SETTINGS
@given(st.integers())
def test_equality_colliding_axis_labels_are_always_rejected(label: int) -> None:
    with pytest.raises(LabelValidationError, match="duplicate or equality-colliding"):
        normalize_axis((label, label), name="generated.axis")


@pytest.mark.parametrize("label", [math.nan, {"nested": math.nan}])
def test_nan_is_rejected_at_any_label_depth(label: object) -> None:
    with pytest.raises(LabelValidationError, match="NaN is not a valid label"):
        freeze_label(label)


@PROPERTY_SETTINGS
@given(st.floats(min_value=-100.0, max_value=-1e-6, allow_nan=False, allow_infinity=False))
def test_negative_prior_mass_is_always_rejected(negative_mass: float) -> None:
    with pytest.raises(PriorValidationError, match="probability cannot be negative"):
        JointPrior.from_mapping(
            {"bad": negative_mass, "good": 1.0 - negative_mass},
            agent_order=("agent",),
            type_spaces={"agent": ("bad", "good")},
        )


@PROPERTY_SETTINGS
@given(st.floats(min_value=0.01, max_value=0.9, allow_nan=False, allow_infinity=False))
def test_nonunit_prior_total_is_always_rejected(total: float) -> None:
    with pytest.raises(PriorValidationError, match="probabilities must sum to one"):
        JointPrior.from_mapping(
            {"left": total, "right": 0.0},
            agent_order=("agent",),
            type_spaces={"agent": ("left", "right")},
        )


@pytest.mark.parametrize("invalid_mass", [math.nan, math.inf, -math.inf])
def test_nonfinite_prior_mass_is_rejected(invalid_mass: float) -> None:
    with pytest.raises(PriorValidationError, match="probability must be finite"):
        JointPrior.from_mapping(
            {"left": invalid_mass, "right": 1.0},
            agent_order=("agent",),
            type_spaces={"agent": ("left", "right")},
        )
