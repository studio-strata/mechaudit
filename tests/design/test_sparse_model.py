from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from mechaudit.design.linear import LinearExpression
from mechaudit.design.sparse_model import (
    ColumnRecord,
    CompiledLP,
    RowRecord,
    RowScale,
    RowSense,
    SparseRow,
)


def _columns() -> tuple[ColumnRecord, ColumnRecord]:
    return (
        ColumnRecord(key="x", block="choice", metadata={"index": 0}),
        ColumnRecord(key="y", block="choice", metadata={"index": 1}),
    )


def test_solver_data_scales_rows_but_preserves_original_economic_units() -> None:
    rows = (
        RowRecord(
            key="minimum_bundle",
            family="feasibility",
            sense=RowSense.GREATER_EQUAL,
            scale=RowScale(
                factor=2.0,
                report_factor=0.5,
                original_unit="joint_utility",
                report_unit="conditional_utility",
            ),
        ),
        RowRecord(
            key="balance",
            family="accounting",
            sense=RowSense.EQUAL,
            scale=RowScale(factor=3.0),
        ),
    )
    model = CompiledLP(
        [[1.0, 2.0], [-1.0, 1.0]],
        [3.0, 0.0],
        [1.0, 0.0],
        rows=rows,
        columns=_columns(),
    )

    np.testing.assert_array_equal(model.matrix.toarray(), [[1.0, 2.0], [-1.0, 1.0]])
    np.testing.assert_array_equal(model.rhs, [3.0, 0.0])
    solver = model.solver_data()
    np.testing.assert_array_equal(solver.matrix.toarray(), [[2.0, 4.0], [-3.0, 3.0]])
    np.testing.assert_array_equal(solver.row_lower, [6.0, 0.0])
    assert np.isposinf(solver.row_upper[0])
    assert solver.row_upper[1] == 0.0

    scale = rows[0].scale
    assert scale.original_slack_to_solver(1.25) == pytest.approx(2.5)
    assert scale.solver_slack_to_original(2.5) == pytest.approx(1.25)
    assert scale.original_slack_to_reported(1.25) == pytest.approx(0.625)
    assert scale.solver_dual_to_reported(0.25) == pytest.approx(1.0)
    assert not rows[1].classifiable


def test_independent_primal_residuals_cover_rows_bounds_and_inactive_rows() -> None:
    model = CompiledLP(
        [[1.0, 1.0], [1.0, -1.0]],
        [3.0, 0.0],
        [0.0, 0.0],
        lower_bounds=[0.0, -np.inf],
        upper_bounds=[np.inf, 2.0],
        rows=(
            RowRecord(key="sum", family="test"),
            RowRecord(key="equal", family="test", sense=RowSense.EQUAL),
        ),
        columns=_columns(),
    )

    residuals = model.evaluate_primal([1.0, 1.0])
    np.testing.assert_array_equal(residuals.row_activity, [2.0, 0.0])
    np.testing.assert_array_equal(residuals.row_slack, [-1.0, 0.0])
    np.testing.assert_array_equal(residuals.row_violation, [1.0, 0.0])
    np.testing.assert_array_equal(residuals.lower_bound_violation, [0.0, 0.0])
    np.testing.assert_array_equal(residuals.upper_bound_violation, [0.0, 0.0])
    assert residuals.max_row_violation == 1.0
    assert residuals.max_bound_violation == 0.0
    assert residuals.max_violation == 1.0
    assert not residuals.is_feasible(0.5)
    assert residuals.is_feasible(1.0)
    assert not residuals.row_activity.flags.writeable

    violated = model.evaluate_primal([-1.0, 3.0])
    np.testing.assert_array_equal(violated.row_activity, [2.0, -4.0])
    np.testing.assert_array_equal(violated.row_violation, [1.0, 4.0])
    np.testing.assert_array_equal(violated.lower_bound_violation, [1.0, 0.0])
    np.testing.assert_array_equal(violated.upper_bound_violation, [0.0, 1.0])
    assert violated.max_violation == 4.0

    inactive_equality = model.evaluate_primal([1.0, 1.0], active_rows=[True, False])
    np.testing.assert_array_equal(inactive_equality.row_violation, [1.0, 0.0])
    np.testing.assert_array_equal(inactive_equality.active_rows, [True, False])


def test_copy_and_fingerprint_are_stable_but_semantic_keys_are_hashed() -> None:
    rows = (RowRecord(key="cap", family="capacity", metadata={"display": "cap"}),)
    columns = _columns()
    model = CompiledLP(
        sparse.coo_matrix(([1.0, -2.0, 2.0], ([0, 0, 0], [0, 1, 1])), shape=(1, 2)),
        [1.0],
        [1.0, 0.0],
        objective_offset=-0.0,
        lower_bounds=[-0.0, -np.inf],
        upper_bounds=[2.0, np.inf],
        rows=rows,
        columns=columns,
    )
    # Duplicate y coefficients cancel during sparse canonicalization.
    assert model.nnz == 1
    assert model.fingerprint == model.copy().fingerprint
    assert (
        model.fingerprint
        == CompiledLP(
            [[1.0, 0.0]],
            [1.0],
            [1.0, 0.0],
            lower_bounds=[0.0, -np.inf],
            upper_bounds=[2.0, np.inf],
            rows=rows,
            columns=columns,
        ).fingerprint
    )
    assert len(model.fingerprint) == 64
    with pytest.raises(ValueError):
        model.objective[0] = 4.0

    changed_row_key = CompiledLP(
        model.matrix,
        model.rhs,
        model.objective,
        lower_bounds=model.lower_bounds,
        upper_bounds=model.upper_bounds,
        rows=(RowRecord(key="different-cap", family="capacity"),),
        columns=columns,
    )
    changed_column_key = CompiledLP(
        model.matrix,
        model.rhs,
        model.objective,
        lower_bounds=model.lower_bounds,
        upper_bounds=model.upper_bounds,
        rows=rows,
        columns=(ColumnRecord(key="different-x", block="choice"), columns[1]),
    )
    changed_objective = model.with_objective([0.0, 1.0], offset=2.0)
    assert changed_row_key.fingerprint != model.fingerprint
    assert changed_column_key.fingerprint != model.fingerprint
    assert changed_objective.fingerprint != model.fingerprint
    np.testing.assert_array_equal(model.objective, [1.0, 0.0])
    assert model.objective_offset == 0.0


def test_extra_face_row_and_structural_row_removal_preserve_stable_records() -> None:
    model = CompiledLP(
        [[-1.0, 0.0], [0.0, -1.0]],
        [-2.0, -3.0],
        [1.0, 1.0],
        lower_bounds=[0.0, 0.0],
        rows=(
            RowRecord(key="x_cap", family="capacity"),
            RowRecord(key="y_cap", family="capacity"),
        ),
        columns=_columns(),
    )
    face = SparseRow(
        LinearExpression.from_mapping({0: 1.0, 1: 1.0}, constant=0.1),
        rhs=4.6,
        record=RowRecord(
            key="optimal_face",
            family="optimal_face",
            classifiable=False,
        ),
    )
    expanded = model.with_extra_rows((face,))
    assert expanded.n_rows == 3
    assert expanded.row_index("optimal_face") == 2
    assert expanded.rhs[2] == pytest.approx(4.5)
    np.testing.assert_array_equal(expanded.matrix.toarray()[2], [1.0, 1.0])
    assert expanded.fingerprint != model.fingerprint

    removed = expanded.with_rows_removed(("x_cap", 2))
    assert removed.n_rows == 1
    assert tuple(row.key for row in removed.rows) == ("y_cap",)
    assert removed.row_index("y_cap") == 0
    assert removed.columns == expanded.columns
    assert expanded.n_rows == 3

    assert expanded.with_extra_rows(()) is expanded
    assert expanded.with_rows_removed(()) is expanded
    with pytest.raises(ValueError, match="duplicate row key"):
        model.with_extra_rows(
            (
                SparseRow(
                    LinearExpression.from_mapping({0: 1.0}),
                    rhs=0.0,
                    record=RowRecord(key="x_cap", family="duplicate"),
                ),
            )
        )
