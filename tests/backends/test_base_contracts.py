"""Portable backend configuration and result contracts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pytest

from mechaudit.backends.base import (
    BackendSession,
    BasisSnapshot,
    ModelStatus,
    SolverConfig,
    SolveResult,
)
from mechaudit.design.sparse_model import (
    ColumnRecord,
    CompiledLP,
    RowRecord,
    SparseRow,
)


def _model() -> CompiledLP:
    return CompiledLP(
        [[1.0]],
        [0.0],
        [1.0],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
        rows=(RowRecord("floor", "test"),),
        columns=(ColumnRecord("x", "test"),),
    )


@pytest.mark.parametrize(
    "status",
    (
        ModelStatus.TIME_LIMIT,
        ModelStatus.ITERATION_LIMIT,
        ModelStatus.OBJECTIVE_LIMIT,
        ModelStatus.SOLUTION_LIMIT,
        ModelStatus.INTERRUPTED,
        ModelStatus.MEMORY_LIMIT,
    ),
)
def test_model_status_limit_classification(status: ModelStatus) -> None:
    assert status.is_limit
    assert not ModelStatus.OPTIMAL.is_limit


def test_solver_config_canonicalization_and_overrides() -> None:
    config = SolverConfig(extra_options={"log_to_console": False, "mip_max_nodes": 3, "alpha": 0.5})

    assert config.option_items == (
        ("alpha", 0.5),
        ("log_to_console", False),
        ("mip_max_nodes", 3),
    )
    assert SolverConfig(extra_options=None).option_items == ()
    changed = config.with_overrides(solver="ipm", threads=2)
    assert changed.solver == "ipm"
    assert changed.threads == 2
    assert config.solver == "simplex"


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"solver": "mip"}, ValueError, "solver"),
        ({"presolve": "maybe"}, ValueError, "presolve"),
        ({"run_crossover": "maybe"}, ValueError, "run_crossover"),
        ({"threads": True}, TypeError, "threads"),
        ({"threads": 1.5}, TypeError, "threads"),
        ({"threads": 0}, ValueError, "threads"),
        ({"random_seed": True}, TypeError, "random_seed"),
        ({"random_seed": -1}, ValueError, "random_seed"),
        ({"primal_feasibility_tolerance": 1e-12}, ValueError, "primal"),
        ({"dual_feasibility_tolerance": float("nan")}, ValueError, "dual"),
        ({"ipm_optimality_tolerance": 0.0}, ValueError, "ipm"),
        ({"small_matrix_value": 1e-14}, ValueError, "small_matrix_value"),
        ({"time_limit": -1.0}, ValueError, "time_limit"),
        ({"time_limit": float("inf")}, ValueError, "time_limit"),
        ({"extra_options": {"": 1}}, TypeError, "option names"),
        ({"extra_options": {"x": float("inf")}}, ValueError, "finite"),
        ({"extra_options": {"x": []}}, TypeError, "unsupported value"),
        ({"extra_options": (("x", 1), ("x", 2))}, ValueError, "duplicate"),
    ),
)
def test_solver_config_rejects_invalid_portable_options(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        SolverConfig(**kwargs)  # type: ignore[arg-type]


def test_basis_and_solve_result_validation_and_read_only_arrays() -> None:
    model = _model()
    BasisSnapshot((0,), (0,)).validate_for(model)
    with pytest.raises(ValueError, match="column count"):
        BasisSnapshot((), (0,)).validate_for(model)
    with pytest.raises(ValueError, match="row count"):
        BasisSnapshot((0,), ()).validate_for(model)

    result = SolveResult(
        status=ModelStatus.OPTIMAL,
        solver_name="test",
        solver_version="1",
        message="ok",
        objective_value=1,
        objective_bound=1,
        solver_objective_value=-1,
        primal=[-0.0],
        row_dual=[2.0],
        solver_row_dual=[-2.0],
        solver_column_dual=[0.0],
        active_rows=[True],
        primal_solution_valid=True,
        dual_solution_valid=True,
        run_time_seconds=2,
        solver_info=(("iterations", 3),),
    )

    assert result.is_optimal
    assert result.has_primal_solution
    assert result.has_dual_solution
    assert result.objective_value == 1.0
    assert result.run_time_seconds == 2.0
    assert result.primal is not None
    assert np.signbit(result.primal[0]) is np.False_
    assert not result.primal.flags.writeable
    assert result.active_rows is not None and not result.active_rows.flags.writeable
    assert SolveResult(ModelStatus.ERROR, "x", "1", "bad").has_primal_solution is False
    assert SolveResult(ModelStatus.ERROR, "x", "1", "bad").has_dual_solution is False

    with pytest.raises(ValueError, match="one-dimensional"):
        SolveResult(ModelStatus.ERROR, "x", "1", "bad", primal=[[1.0]])
    with pytest.raises(ValueError, match="one-dimensional"):
        SolveResult(ModelStatus.ERROR, "x", "1", "bad", active_rows=[[True]])


class _RecordingSession(BackendSession):
    def __init__(self, model: CompiledLP) -> None:
        self._model = model
        self._active = np.ones(model.n_rows, dtype=np.bool_)
        self.closed = False
        self.restored: list[tuple[int, ...]] = []

    @property
    def model(self) -> CompiledLP:
        return self._model

    @property
    def active_rows(self) -> np.ndarray:
        return self._active.copy()

    def solve(self, *, basis: BasisSnapshot | None = None) -> SolveResult:
        del basis
        return SolveResult(ModelStatus.EMPTY, "dummy", "1", "empty")

    def set_objective(self, objective: Sequence[float], *, offset: float = 0.0) -> None:
        del objective, offset

    def reset_objective(self) -> None:
        return None

    def remove_rows(self, rows: Iterable[int | str]) -> tuple[int, ...]:
        indices = tuple(int(row) for row in rows)
        for index in indices:
            self._active[index] = False
        return indices

    def restore_rows(self, rows: Iterable[int | str]) -> tuple[int, ...]:
        indices = tuple(int(row) for row in rows)
        for index in indices:
            self._active[index] = True
        self.restored.append(indices)
        return indices

    def add_rows(self, rows: Sequence[SparseRow]) -> tuple[int, ...]:
        return tuple(range(self.model.n_rows, self.model.n_rows + len(rows)))

    def close(self) -> None:
        self.closed = True


def test_backend_session_contexts_restore_rows_and_close() -> None:
    session = _RecordingSession(_model())

    with pytest.raises(RuntimeError, match="boom"), session.rows_removed((0,)) as removed:
        assert removed == (0,)
        assert not session.active_rows[0]
        raise RuntimeError("boom")
    assert session.active_rows[0]
    assert session.restored == [(0,)]

    with session as entered:
        assert entered is session
        assert entered.solve().status is ModelStatus.EMPTY
    assert session.closed
