from __future__ import annotations

import numpy as np
import pytest

from mechaudit.backends.base import BackendFeatureError, ModelStatus, SolverConfig
from mechaudit.backends.highs import HighsBackend
from mechaudit.design.linear import LinearExpression
from mechaudit.design.sparse_model import (
    ColumnRecord,
    CompiledLP,
    RowRecord,
    RowScale,
    RowSense,
    SparseRow,
)


def _column(key: str = "x") -> tuple[ColumnRecord, ...]:
    return (ColumnRecord(key=key, block="choice"),)


def test_canonical_greater_equal_dual_sign_and_report_scaling() -> None:
    model = CompiledLP(
        [[-1.0]],
        [-2.0],
        [1.0],
        lower_bounds=[0.0],
        rows=(
            RowRecord(
                key="x_cap",
                family="capacity",
                scale=RowScale(factor=5.0, report_factor=2.0),
            ),
        ),
        columns=_column(),
    )
    result = HighsBackend().solve(
        model,
        SolverConfig(solver="simplex", presolve="off", output=False),
    )

    assert result.status is ModelStatus.OPTIMAL
    assert result.has_primal_solution
    assert result.has_dual_solution
    np.testing.assert_allclose(result.primal, [2.0], atol=1e-9)
    assert result.objective_value == pytest.approx(2.0)
    assert result.solver_objective_value == pytest.approx(2.0)
    assert result.solver_row_dual is not None
    assert result.row_dual is not None
    # HiGHS uses a nonpositive row dual for a maximization lower-bound row.
    assert result.solver_row_dual[0] == pytest.approx(-0.2)
    # Canonical multiplier is positive; 5 / 2 converts solver to report units.
    assert result.row_dual[0] == pytest.approx(0.5)
    assert result.row_dual[0] >= 0.0
    assert result.residuals is not None
    assert result.residuals.max_violation <= 1e-9
    assert result.residuals.row_slack[0] == pytest.approx(0.0)


def test_equality_row_solves_and_has_independently_verified_residual() -> None:
    equality = RowRecord(
        key="fix_x",
        family="accounting",
        sense=RowSense.EQUAL,
        scale=RowScale(factor=4.0),
    )
    model = CompiledLP(
        [[1.0]],
        [1.0],
        [1.0],
        lower_bounds=[-10.0],
        upper_bounds=[10.0],
        rows=(equality,),
        columns=_column(),
    )
    result = HighsBackend().solve(model, SolverConfig(presolve="off"))

    assert result.status is ModelStatus.OPTIMAL
    np.testing.assert_allclose(result.primal, [1.0], atol=1e-9)
    assert result.objective_value == pytest.approx(1.0)
    assert not equality.classifiable
    assert result.residuals is not None
    np.testing.assert_allclose(result.residuals.row_activity, [1.0], atol=1e-9)
    np.testing.assert_allclose(result.residuals.row_slack, [0.0], atol=1e-9)
    np.testing.assert_allclose(result.residuals.row_violation, [0.0], atol=1e-9)
    assert result.residuals.max_violation <= 1e-9


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            CompiledLP(
                [[1.0]],
                [1.0],
                [1.0],
                lower_bounds=[0.0],
                upper_bounds=[0.0],
                rows=(RowRecord(key="x_at_least_one", family="test"),),
                columns=_column(),
            ),
            ModelStatus.INFEASIBLE,
        ),
        (
            CompiledLP(
                np.empty((0, 1)),
                [],
                [1.0],
                lower_bounds=[0.0],
                upper_bounds=[np.inf],
                rows=(),
                columns=_column(),
            ),
            ModelStatus.UNBOUNDED,
        ),
    ],
)
def test_infeasible_and_unbounded_statuses_are_normalized(
    model: CompiledLP,
    expected: ModelStatus,
) -> None:
    result = HighsBackend().solve(model)
    assert result.status is expected
    assert not result.is_optimal
    assert result.objective_bound is None
    assert result.solver_name == "highs"
    assert result.solver_version
    assert result.message


def test_ipm_without_crossover_returns_a_central_optimal_candidate() -> None:
    model = CompiledLP(
        [[-1.0, -1.0]],
        [-1.0],
        [1.0, 1.0],
        lower_bounds=[0.0, 0.0],
        rows=(RowRecord(key="simplex_cap", family="capacity"),),
        columns=(
            ColumnRecord(key="x", block="choice"),
            ColumnRecord(key="y", block="choice"),
        ),
    )
    result = HighsBackend().solve(
        model,
        SolverConfig(
            solver="ipm",
            presolve="off",
            run_crossover="off",
            output=False,
        ),
    )

    assert result.status is ModelStatus.OPTIMAL
    assert result.ipm_iterations > 0
    assert result.crossover_iterations == 0
    assert result.primal is not None
    assert np.sum(result.primal) == pytest.approx(1.0, abs=1e-8)
    assert np.all(result.primal > 0.0)
    assert result.residuals is not None
    assert result.residuals.max_violation <= 1e-8


def test_session_objective_mutation_and_reset_restore_model_and_value() -> None:
    model = CompiledLP(
        np.empty((0, 2)),
        [],
        [1.0, 0.0],
        objective_offset=2.0,
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
        rows=(),
        columns=(
            ColumnRecord(key="x", block="choice"),
            ColumnRecord(key="y", block="choice"),
        ),
    )
    initial_fingerprint = model.fingerprint

    with HighsBackend().session(model, SolverConfig(presolve="off")) as session:
        initial = session.solve()
        assert initial.status is ModelStatus.OPTIMAL
        assert initial.objective_value == pytest.approx(3.0)
        assert initial.solver_objective_value == pytest.approx(3.0)

        session.set_objective([0.0, 1.0], offset=5.0)
        assert session.model.fingerprint != initial_fingerprint
        np.testing.assert_array_equal(session.model.objective, [0.0, 1.0])
        assert session.model.objective_offset == 5.0
        mutated = session.solve()
        assert mutated.status is ModelStatus.OPTIMAL
        assert mutated.objective_value == pytest.approx(6.0)
        assert mutated.solver_objective_value == pytest.approx(6.0)

        session.reset_objective()
        assert session.model.fingerprint == initial_fingerprint
        np.testing.assert_array_equal(session.model.objective, [1.0, 0.0])
        assert session.model.objective_offset == 2.0
        restored = session.solve()
        assert restored.objective_value == pytest.approx(3.0)
        assert restored.solver_objective_value == pytest.approx(3.0)


def test_logical_row_removal_and_restore_keep_stable_row_indices() -> None:
    model = CompiledLP(
        [[-1.0]],
        [-2.0],
        [1.0],
        lower_bounds=[0.0],
        upper_bounds=[10.0],
        rows=(RowRecord(key="x_cap", family="capacity"),),
        columns=_column(),
    )

    with HighsBackend().session(model, SolverConfig(presolve="off")) as session:
        constrained = session.solve()
        np.testing.assert_allclose(constrained.primal, [2.0], atol=1e-9)
        original_fingerprint = session.model.fingerprint

        assert session.remove_rows(("x_cap", "x_cap")) == (0,)
        np.testing.assert_array_equal(session.active_rows, [False])
        assert session.model.fingerprint == original_fingerprint
        relaxed = session.solve()
        np.testing.assert_allclose(relaxed.primal, [10.0], atol=1e-9)
        assert relaxed.active_rows is not None
        np.testing.assert_array_equal(relaxed.active_rows, [False])
        assert relaxed.residuals is not None
        assert relaxed.residuals.row_slack[0] == pytest.approx(-8.0)
        assert relaxed.residuals.row_violation[0] == 0.0
        assert relaxed.residuals.max_violation == 0.0
        assert relaxed.row_dual is not None
        assert np.isnan(relaxed.row_dual[0])

        assert session.restore_rows((0,)) == (0,)
        np.testing.assert_array_equal(session.active_rows, [True])
        restored = session.solve()
        np.testing.assert_allclose(restored.primal, [2.0], atol=1e-9)
        assert restored.residuals is not None
        assert restored.residuals.max_violation <= 1e-9


def test_appended_optimal_face_row_can_drive_an_auxiliary_objective() -> None:
    model = CompiledLP(
        np.empty((0, 1)),
        [],
        [1.0],
        lower_bounds=[0.0],
        upper_bounds=[2.0],
        rows=(),
        columns=_column(),
    )
    face = SparseRow(
        expression=LinearExpression.from_mapping({0: 1.0}),
        rhs=1.9,
        record=RowRecord(
            key="objective_face",
            family="optimal_face",
            classifiable=False,
        ),
    )

    with HighsBackend().session(model, SolverConfig(presolve="off")) as session:
        base = session.solve()
        assert base.objective_value == pytest.approx(2.0)
        assert session.add_rows((face,)) == (0,)
        assert session.model.n_rows == 1
        assert session.model.row_index("objective_face") == 0
        np.testing.assert_array_equal(session.active_rows, [True])

        session.set_objective([-1.0])
        auxiliary = session.solve()
        assert auxiliary.status is ModelStatus.OPTIMAL
        np.testing.assert_allclose(auxiliary.primal, [1.9], atol=1e-8)
        assert auxiliary.objective_value == pytest.approx(-1.9)
        assert auxiliary.residuals is not None
        assert auxiliary.residuals.row_slack[0] == pytest.approx(0.0, abs=1e-8)
        assert auxiliary.residuals.max_violation <= 1e-8


def test_scaled_coefficient_at_highs_drop_threshold_is_rejected_deterministically() -> None:
    model = CompiledLP(
        [[1e-12]],
        [0.0],
        [0.0],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
        rows=(RowRecord(key="tiny", family="numerical"),),
        columns=_column(),
    )
    with pytest.raises(BackendFeatureError, match=r"at or below.*small_matrix_value"):
        HighsBackend().session(
            model,
            SolverConfig(small_matrix_value=1e-12),
        )
