"""Process-worker correctness and serialization contracts."""

from __future__ import annotations

import pickle

import pytest

from mechaudit import (
    DSIC,
    ExpectedRevenue,
    ExPostIR,
    FiniteEnvironment,
    NoSubsidies,
    PrivateValues,
)
from mechaudit.design.sparse_model import CompiledLP
from mechaudit.optimality.tasks import _resolve_workers


def _solution():
    environment = FiniteEnvironment(
        agents={"buyer": ("low", "high")},
        outcomes=("no_sale", "sale"),
        prior={"low": 0.5, "high": 0.5},
        values=PrivateValues(
            lambda _agent, own_type, outcome: (
                {"low": 1.0, "high": 2.0}[own_type] if outcome == "sale" else 0.0
            )
        ),
    )
    return environment.design(
        objective=ExpectedRevenue(),
        requirements=(DSIC(), ExPostIR(), NoSubsidies()),
    ).solve()


def _activity_signature(report):
    return tuple(
        (
            item.constraint_id,
            item.classification,
            item.minimum_slack,
            item.maximum_slack,
            item.min_status,
            item.max_status,
        )
        for item in report.constraints
    )


def _dual_signature(report):
    return tuple(
        (item.constraint_id, item.lower, item.upper, item.relevance) for item in report.dual_ranges
    )


def test_compiled_lp_pickle_round_trip_preserves_fingerprint_and_units() -> None:
    model = _solution().compiled.model

    restored = pickle.loads(pickle.dumps(model))

    assert isinstance(restored, CompiledLP)
    assert restored is not model
    assert restored.fingerprint == model.fingerprint
    assert restored.rows == model.rows
    assert restored.columns == model.columns
    assert restored.matrix.toarray() == pytest.approx(model.matrix.toarray())
    assert restored.rhs == pytest.approx(model.rhs)


@pytest.mark.slow
def test_process_workers_match_serial_activity_and_dual_ranges() -> None:
    solution = _solution()

    serial = solution.analyze_optima(level="full", objective_delta=0.0, dual_ranges=True, n_jobs=1)
    parallel = solution.analyze_optima(
        level="full", objective_delta=0.0, dual_ranges=True, n_jobs=2
    )

    assert not serial.failures
    assert not parallel.failures
    assert parallel.solve_counts == serial.solve_counts
    assert _activity_signature(parallel) == _activity_signature(serial)
    assert _dual_signature(parallel) == _dual_signature(serial)


def test_automatic_worker_resolution_is_bounded_and_avoids_tiny_batches() -> None:
    assert _resolve_workers(-1, 0) == 0
    assert _resolve_workers(-1, 1) == 1
    assert _resolve_workers(-1, 7) == 1
    assert _resolve_workers(-1, 64) == 1
    assert 1 <= _resolve_workers(-1, 1024) <= 4
    assert _resolve_workers(2, 4) in {1, 2}
    with pytest.raises(ValueError, match="n_jobs"):
        _resolve_workers(0, 4)
    with pytest.raises(ValueError, match="task_count"):
        _resolve_workers(1, -1)
