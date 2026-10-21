from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mechaudit.core.environment import FiniteEnvironment
from mechaudit.core.valuations import PrivateValues
from mechaudit.exceptions import DomainMismatchError, ValidationError
from mechaudit.mechanisms.direct import DirectMechanism, MechanismTables


def _one_buyer_environment() -> FiniteEnvironment:
    return FiniteEnvironment(
        agents={"buyer": ("low", "high")},
        outcomes=("no_sale", "sale"),
        values=PrivateValues(
            lambda agent, type_value, outcome: (
                {"low": 1.0, "high": 2.0}[type_value]
                if agent == "buyer" and outcome == "sale"
                else 0.0
            )
        ),
    )


def _two_agent_environment() -> FiniteEnvironment:
    return FiniteEnvironment(
        agents={"buyer": ("low", "high"), "seller": ("low", "high")},
        outcomes=("no_trade", "trade"),
        values=PrivateValues(lambda agent, type_value, outcome: 0.0),
    )


def test_nested_mappings_bind_one_agent_shorthands_and_round_trip() -> None:
    environment = _one_buyer_environment()
    raw = DirectMechanism(
        allocation={
            "low": {"no_sale": 0.0, "sale": 1.0},
            "high": {"no_sale": 0.0, "sale": 1.0},
        },
        payments={"low": 1.0, "high": 1.0},
    )
    assert not raw.is_bound
    with pytest.raises(ValidationError, match="not bound"):
        _ = raw.allocation
    with pytest.raises(ValidationError, match="not bound"):
        _ = raw.fingerprint

    mechanism = raw.bind(environment)
    np.testing.assert_array_equal(mechanism.allocation, [[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_array_equal(mechanism.payments, [[1.0], [1.0]])
    assert mechanism.allocation.dtype == np.float64
    assert mechanism.payments.dtype == np.float64
    assert not mechanism.allocation.flags.writeable
    assert not mechanism.payments.flags.writeable
    with pytest.raises(ValueError):
        mechanism.allocation.flags.writeable = True
    assert mechanism.bind(environment) is mechanism

    exported = mechanism.to_dict()
    restored = DirectMechanism.from_mapping(exported, environment=environment)
    np.testing.assert_array_equal(restored.allocation, mechanism.allocation)
    np.testing.assert_array_equal(restored.payments, mechanism.payments)
    assert restored.domain_fingerprint == environment.domain_fingerprint
    assert restored.fingerprint == mechanism.fingerprint
    assert len(mechanism.fingerprint) == 64
    assert mechanism.fingerprint != mechanism.domain_fingerprint


def test_row_and_cell_callbacks_are_evaluated_on_the_canonical_domain() -> None:
    environment = _one_buyer_environment()
    row_calls: list[tuple[str, ...]] = []

    def allocation_row(report: object) -> dict[str, float]:
        row_calls.append(tuple(report.values_tuple))  # type: ignore[attr-defined]
        return {"no_sale": 0.0, "sale": 1.0}

    row_mechanism = DirectMechanism.from_functions(
        allocation_row,
        lambda report: 1.0,
    ).bind(environment)
    np.testing.assert_array_equal(row_mechanism.allocation, [[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_array_equal(row_mechanism.payments, [[1.0], [1.0]])
    assert row_calls == [("low",), ("high",)]

    cell_mechanism = DirectMechanism.from_functions(
        lambda report, outcome: 1.0 if outcome == "sale" else 0.0,
        lambda report, agent: 2.0 if report[agent] == "high" else 1.0,
        environment=environment,
    )
    np.testing.assert_array_equal(cell_mechanism.allocation, [[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_array_equal(cell_mechanism.payments, [[1.0], [2.0]])


def test_labeled_arrays_reorder_every_axis_and_numeric_fingerprint_changes() -> None:
    environment = _two_agent_environment()
    allocation = np.arange(8.0).reshape(4, 2)
    payments = np.arange(8.0, 16.0).reshape(4, 2)

    mechanism = DirectMechanism.from_arrays(
        allocation[::-1, ::-1],
        payments[::-1, ::-1],
        report_profiles=environment.report_profiles[::-1],
        payment_report_profiles=environment.report_profiles[::-1],
        outcomes=environment.outcomes[::-1],
        agents=environment.agents[::-1],
        environment=environment,
    )
    np.testing.assert_array_equal(mechanism.allocation, allocation)
    np.testing.assert_array_equal(mechanism.payments, payments)

    changed = DirectMechanism.from_arrays(
        allocation,
        payments + np.eye(4, 2),
        environment=environment,
    )
    assert changed.domain_fingerprint == mechanism.domain_fingerprint
    assert changed.fingerprint != mechanism.fingerprint


def test_pandas_tables_use_labeled_multiindex_and_round_trip() -> None:
    environment = _two_agent_environment()
    mechanism = DirectMechanism.from_arrays(
        [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75], [0.5, 0.5]],
        [[0.0, 0.0], [1.0, -1.0], [0.5, -0.5], [2.0, -1.0]],
        environment=environment,
    )

    tables = mechanism.to_pandas()
    assert isinstance(tables, MechanismTables)
    assert isinstance(tables.allocation, pd.DataFrame)
    assert isinstance(tables.allocation.index, pd.MultiIndex)
    assert tables.allocation.index.names == ["buyer", "seller"]
    assert tables.allocation.columns.tolist() == ["no_trade", "trade"]
    assert tables.payments.columns.tolist() == ["buyer", "seller"]
    assert tables.allocation.attrs["domain_fingerprint"] == environment.domain_fingerprint
    assert tables.payments.attrs["mechanism_fingerprint"] == mechanism.fingerprint

    restored = DirectMechanism.from_tables(
        tables.allocation.iloc[::-1, ::-1],
        tables.payments.iloc[::-1, ::-1],
        environment=environment,
    )
    np.testing.assert_array_equal(restored.allocation, mechanism.allocation)
    np.testing.assert_array_equal(restored.payments, mechanism.payments)
    assert restored.fingerprint == mechanism.fingerprint


def test_import_retains_economically_invalid_allocations_for_the_auditor() -> None:
    environment = _one_buyer_environment()
    mechanism = DirectMechanism.from_mapping(
        {
            "low": {"no_sale": -0.25, "sale": 1.25},
            "high": {"no_sale": 0.0, "sale": 0.75},
        },
        {"low": 0.0, "high": 0.0},
        environment=environment,
    )
    np.testing.assert_array_equal(
        mechanism.allocation,
        [[-0.25, 1.25], [0.0, 0.75]],
    )


@pytest.mark.parametrize(
    ("allocation", "payments", "error"),
    [
        (
            {"low": {"no_sale": np.nan, "sale": 1.0}, "high": {"no_sale": 0.0, "sale": 1.0}},
            {"low": 0.0, "high": 0.0},
            "finite",
        ),
        (
            {"low": {"no_sale": 0.0, "sale": 1.0}},
            {"low": 0.0},
            "every report profile",
        ),
        (
            {
                "low": {"no_sale": 0.0, "unknown": 1.0},
                "high": {"no_sale": 0.0, "unknown": 1.0},
            },
            {"low": 0.0, "high": 0.0},
            "unknown outcome",
        ),
    ],
)
def test_mapping_import_rejects_nonfinite_or_misaligned_data(
    allocation: dict[object, object],
    payments: dict[object, object],
    error: str,
) -> None:
    environment = _one_buyer_environment()
    if error == "finite":
        with pytest.raises(ValidationError, match=error):
            DirectMechanism(allocation=allocation, payments=payments)
    else:
        raw = DirectMechanism(allocation=allocation, payments=payments)
        with pytest.raises(ValidationError, match=error):
            raw.bind(environment)


def test_array_shape_and_domain_fingerprint_mismatches_are_rejected() -> None:
    environment = _one_buyer_environment()
    with pytest.raises(ValidationError, match="shape does not match"):
        DirectMechanism.from_arrays([[1.0, 0.0]], [0.0], environment=environment)

    unbound = DirectMechanism.from_arrays(
        [[1.0, 0.0], [0.0, 1.0]],
        [0.0, 1.0],
        domain_fingerprint="not-the-environment-domain",
    )
    with pytest.raises(DomainMismatchError, match="fingerprint"):
        unbound.bind(environment)
