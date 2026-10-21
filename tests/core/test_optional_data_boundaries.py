"""Regression tests for optional labeled-data and serialization boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mechaudit._serialization import SCHEMA_VERSION, dumps_report
from mechaudit._version import __version__
from mechaudit.audit.engine import audit_mechanism
from mechaudit.core.environment import FiniteEnvironment
from mechaudit.core.labels import StructuredLabel
from mechaudit.core.prior import JointPrior
from mechaudit.core.valuations import PrivateValues
from mechaudit.mechanisms.direct import DirectMechanism
from mechaudit.requirements.participation import ExPostIR


def _zero_values(_agent: object, _type_value: object, _outcome: object) -> float:
    return 0.0


def test_joint_prior_from_xarray_preserves_labels_across_reordered_dimensions() -> None:
    xr = pytest.importorskip("xarray")
    canonical = xr.DataArray(
        [[0.10, 0.20], [0.30, 0.40]],
        dims=("buyer", "seller"),
        coords={"buyer": ["low", "high"], "seller": ["low", "high"]},
    )
    reordered = canonical.sel(
        buyer=["high", "low"],
        seller=["high", "low"],
    ).transpose("seller", "buyer")

    imported = JointPrior.from_xarray(reordered)
    assert imported.agents == ("seller", "buyer")
    assert dict(imported.type_spaces) == {
        "seller": ("high", "low"),
        "buyer": ("high", "low"),
    }

    environment = FiniteEnvironment(
        agents={"buyer": ("low", "high"), "seller": ("low", "high")},
        outcomes=("outcome",),
        prior=imported,
        values=PrivateValues(_zero_values),
    )
    assert environment.prior is not None
    expected = {
        ("low", "low"): 0.10,
        ("low", "high"): 0.20,
        ("high", "low"): 0.30,
        ("high", "high"): 0.40,
    }
    for profile, probability in expected.items():
        assert environment.prior[environment.coerce_profile(profile)] == pytest.approx(probability)


def _two_agent_environment() -> FiniteEnvironment:
    return FiniteEnvironment(
        agents={"buyer": ("low", "high"), "seller": ("low", "high")},
        outcomes=("no_trade", "trade"),
        prior={
            profile: 0.25
            for profile in (
                ("low", "low"),
                ("low", "high"),
                ("high", "low"),
                ("high", "high"),
            )
        },
        values=PrivateValues(_zero_values),
    )


def _reordered_xarray_mechanism_inputs(
    environment: FiniteEnvironment,
) -> tuple[object, object, np.ndarray, np.ndarray]:
    xr = pytest.importorskip("xarray")
    expected_allocation = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
        dtype=float,
    )
    expected_payments = np.array(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        dtype=float,
    )
    metadata = {
        "payment_sign": "agent_to_designer",
        "domain_fingerprint": environment.domain_fingerprint,
    }
    allocation = xr.DataArray(
        expected_allocation.reshape(2, 2, 2),
        dims=("buyer", "seller", "outcome"),
        coords={
            "buyer": ["low", "high"],
            "seller": ["low", "high"],
            "outcome": ["no_trade", "trade"],
        },
        attrs=metadata,
    )
    payments = xr.DataArray(
        expected_payments.reshape(2, 2, 2),
        dims=("buyer", "seller", "agent"),
        coords={
            "buyer": ["low", "high"],
            "seller": ["low", "high"],
            "agent": ["buyer", "seller"],
        },
        attrs=metadata,
    )
    reordered_allocation = allocation.sel(
        buyer=["high", "low"],
        seller=["high", "low"],
        outcome=["trade", "no_trade"],
    ).transpose("outcome", "seller", "buyer")
    reordered_payments = payments.sel(
        buyer=["high", "low"],
        seller=["high", "low"],
        agent=["seller", "buyer"],
    ).transpose("agent", "buyer", "seller")
    return reordered_allocation, reordered_payments, expected_allocation, expected_payments


def test_direct_mechanism_from_reordered_xarray_dataarrays_and_dataset() -> None:
    xr = pytest.importorskip("xarray")
    environment = _two_agent_environment()
    allocation, payments, expected_allocation, expected_payments = (
        _reordered_xarray_mechanism_inputs(environment)
    )

    from_arrays = DirectMechanism.from_xarray(
        allocation,
        payments,
        environment=environment,
        report_dims=("buyer", "seller"),
    )
    np.testing.assert_array_equal(from_arrays.allocation, expected_allocation)
    np.testing.assert_array_equal(from_arrays.payments, expected_payments)

    dataset = xr.Dataset({"allocation": allocation, "payments": payments})
    from_dataset = DirectMechanism.from_xarray(
        dataset,
        environment=environment,
        report_dims=("buyer", "seller"),
    )
    np.testing.assert_array_equal(from_dataset.allocation, expected_allocation)
    np.testing.assert_array_equal(from_dataset.payments, expected_payments)
    assert from_dataset.fingerprint == from_arrays.fingerprint
    assert from_dataset.domain_fingerprint == environment.domain_fingerprint


def _structured_environment_and_mechanism() -> tuple[FiniteEnvironment, DirectMechanism]:
    raw_types = (
        {"quality": "low", "urgency": 1},
        {"quality": "high", "urgency": 2},
    )
    prior = JointPrior.from_records(
        (
            {"buyer": raw_types[0], "probability": 0.4},
            {"buyer": raw_types[1], "probability": 0.6},
        ),
        agent_order=("buyer",),
        type_spaces={"buyer": raw_types},
    )
    environment = FiniteEnvironment(
        agents={"buyer": raw_types},
        outcomes=("no_sale", "sale"),
        prior=prior,
        values=PrivateValues(
            lambda _agent, type_value, outcome: (
                float(type_value["urgency"]) if outcome == "sale" else 0.0
            )
        ),
    )
    low, high = environment.types["buyer"]
    mechanism = DirectMechanism(
        allocation={
            low: {"no_sale": 0.75, "sale": 0.25},
            high: {"no_sale": 0.10, "sale": 0.90},
        },
        payments={low: 0.2, high: 1.1},
        environment=environment,
    )
    return environment, mechanism


def test_structured_type_labels_round_trip_through_prior_environment_and_pandas() -> None:
    environment, mechanism = _structured_environment_and_mechanism()
    assert environment.prior is not None
    assert all(isinstance(type_value, StructuredLabel) for type_value in environment.types["buyer"])
    assert environment.prior.marginal("buyer", {"urgency": 1, "quality": "low"}) == pytest.approx(
        0.4
    )
    assert environment.evaluate_value(
        "buyer",
        {"quality": "high", "urgency": 2},
        "sale",
    ) == pytest.approx(2.0)

    tables = mechanism.to_pandas()
    assert tables.allocation.index.name == "buyer"
    assert list(tables.allocation.index) == list(environment.types["buyer"])
    assert all(isinstance(item, StructuredLabel) for item in tables.allocation.index)
    assert tables.allocation.attrs["domain_fingerprint"] == environment.domain_fingerprint

    restored = DirectMechanism.from_tables(
        tables.allocation.iloc[::-1, ::-1],
        tables.payments.iloc[::-1],
        environment=environment,
    )
    np.testing.assert_array_equal(restored.allocation, mechanism.allocation)
    np.testing.assert_array_equal(restored.payments, mechanism.payments)
    assert restored.fingerprint == mechanism.fingerprint


def test_report_json_envelope_version_and_serialization_are_deterministic(
    tmp_path: Path,
) -> None:
    environment, mechanism = _structured_environment_and_mechanism()
    first = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(ExPostIR(outside_option=0.05),),
    )
    second = audit_mechanism(
        environment=environment,
        mechanism=mechanism,
        requirements=(ExPostIR(outside_option=0.05),),
    )

    serialized = first.to_json()
    assert serialized == second.to_json()
    decoded = json.loads(serialized)
    assert decoded["schema_version"] == SCHEMA_VERSION == "1.0"
    assert decoded["library_version"] == __version__
    assert decoded["report_kind"] == "mechanism_audit"
    assert set(decoded) == {
        "checks",
        "environment_fingerprint",
        "epsilon_ic",
        "feasibility_tolerance",
        "library_version",
        "mechanism_fingerprint",
        "numerical_health",
        "passed",
        "payment_sign_convention",
        "probability_tolerance",
        "report_kind",
        "requirements",
        "schema_version",
    }
    assert list(decoded) == sorted(decoded)

    path = tmp_path / "audit.json"
    first.write_json(path)
    assert path.read_text(encoding="utf-8") == serialized + "\n"

    unordered = json.loads(dumps_report("set_order", {"labels": {"gamma", "alpha", "beta"}}))
    assert unordered["labels"] == ["alpha", "beta", "gamma"]
