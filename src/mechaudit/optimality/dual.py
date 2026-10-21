"""Explicit dual-LP construction and multiplier ranges."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
from scipy import sparse

from mechaudit.backends.base import ModelStatus
from mechaudit.backends.highs import HighsBackend
from mechaudit.design.solution import DesignSolution
from mechaudit.design.sparse_model import (
    ColumnRecord,
    CompiledLP,
    RowRecord,
    RowScale,
    RowSense,
)
from mechaudit.optimality.report import (
    AuxiliaryStatus,
    DualRangeResult,
    DualRelevance,
    TaskFailure,
)
from mechaudit.optimality.tasks import (
    _resolve_workers,
    certified_optimal_result,
    solver_config,
)
from mechaudit.options import TolerancePolicy


@dataclass(frozen=True)
class ExplicitDualLP:
    """A canonical LP representation of the primal problem's dual face."""

    model: CompiledLP
    inequality_columns: dict[int, int]


def build_explicit_dual(
    primal: CompiledLP,
    *,
    optimal_value: float,
    delta: float,
) -> ExplicitDualLP:
    """Build the dual and impose dual objective <= ``optimal_value + delta``.

    For the primal canonical form ``max c'z+c0``, ``Gz>=h``, ``Ez=d``,
    the construction uses nonnegative inequality/bound multipliers and free
    equality multipliers.  Its face row is the negative dual objective:
    ``-D >= -optimal_value + c0 - delta``.
    """

    if delta < 0.0 or not np.isfinite(delta):
        raise ValueError("dual-face delta must be finite and nonnegative")
    ge_rows = [
        index for index, record in enumerate(primal.rows) if record.sense is RowSense.GREATER_EQUAL
    ]
    eq_rows = [index for index, record in enumerate(primal.rows) if record.sense is RowSense.EQUAL]
    finite_lower = [index for index, value in enumerate(primal.lower_bounds) if np.isfinite(value)]
    finite_upper = [index for index, value in enumerate(primal.upper_bounds) if np.isfinite(value)]

    columns: list[ColumnRecord] = []
    lower: list[float] = []
    upper: list[float] = []
    inequality_columns: dict[int, int] = {}
    equality_columns: dict[int, int] = {}
    lower_columns: dict[int, int] = {}
    upper_columns: dict[int, int] = {}

    for row_index in ge_rows:
        column = len(columns)
        inequality_columns[row_index] = column
        source = primal.rows[row_index]
        columns.append(
            ColumnRecord(
                key=f"dual.lambda[{source.key}]",
                block="dual_inequality",
                metadata={"primal_row": row_index, "constraint_id": source.key},
            )
        )
        lower.append(0.0)
        upper.append(np.inf)
    for row_index in eq_rows:
        column = len(columns)
        equality_columns[row_index] = column
        source = primal.rows[row_index]
        columns.append(
            ColumnRecord(
                key=f"dual.nu[{source.key}]",
                block="dual_equality",
                metadata={"primal_row": row_index, "constraint_id": source.key},
            )
        )
        lower.append(-np.inf)
        upper.append(np.inf)
    for primal_column in finite_lower:
        column = len(columns)
        lower_columns[primal_column] = column
        columns.append(
            ColumnRecord(
                key=f"dual.lower[{primal.columns[primal_column].key}]",
                block="dual_lower_bound",
                metadata={"primal_column": primal_column},
            )
        )
        lower.append(0.0)
        upper.append(np.inf)
    for primal_column in finite_upper:
        column = len(columns)
        upper_columns[primal_column] = column
        columns.append(
            ColumnRecord(
                key=f"dual.upper[{primal.columns[primal_column].key}]",
                block="dual_upper_bound",
                metadata={"primal_column": primal_column},
            )
        )
        lower.append(0.0)
        upper.append(np.inf)

    n_dual_columns = len(columns)
    if n_dual_columns == 0:
        # CompiledLP requires a column.  A fixed zero dummy is harmless and
        # keeps dual infeasibility visible through the stationarity equations.
        columns.append(ColumnRecord(key="dual.dummy", block="dual_dummy"))
        lower.append(0.0)
        upper.append(0.0)
        n_dual_columns = 1

    primal_matrix = primal.matrix.tocoo()
    entries: list[tuple[int, int, float]] = []
    for source_row, primal_column, coefficient in zip(
        primal_matrix.row, primal_matrix.col, primal_matrix.data, strict=True
    ):
        source_row_index = int(source_row)
        target_row = int(primal_column)
        if source_row_index in inequality_columns:
            entries.append(
                (
                    target_row,
                    inequality_columns[source_row_index],
                    float(coefficient),
                )
            )
        else:
            entries.append(
                (
                    target_row,
                    equality_columns[source_row_index],
                    float(coefficient),
                )
            )
    for primal_column, column in lower_columns.items():
        entries.append((primal_column, column, 1.0))
    for primal_column, column in upper_columns.items():
        entries.append((primal_column, column, -1.0))

    # q is negative of the dual objective without the primal objective offset.
    q = np.zeros(n_dual_columns, dtype=float)
    for row_index, column in inequality_columns.items():
        q[column] = float(primal.rhs[row_index])
    for row_index, column in equality_columns.items():
        q[column] = float(primal.rhs[row_index])
    for primal_column, column in lower_columns.items():
        q[column] = float(primal.lower_bounds[primal_column])
    for primal_column, column in upper_columns.items():
        q[column] = -float(primal.upper_bounds[primal_column])

    face_row_index = primal.n_columns
    for column, coefficient in enumerate(q):
        if coefficient != 0.0:
            entries.append((face_row_index, column, float(coefficient)))
    if entries:
        rows_index, columns_index, data = zip(*entries, strict=True)
        matrix = sparse.coo_matrix(
            (data, (rows_index, columns_index)),
            shape=(primal.n_columns + 1, n_dual_columns),
            dtype=float,
        ).tocsr()
    else:
        matrix = sparse.csr_matrix((primal.n_columns + 1, n_dual_columns), dtype=float)
    rhs = np.concatenate(
        (
            -np.asarray(primal.objective, dtype=float),
            np.asarray(
                [-float(optimal_value) + primal.objective_offset - float(delta)],
                dtype=float,
            ),
        )
    )
    rows = (
        *(
            RowRecord(
                key=f"dual.stationarity[{primal.columns[index].key}]",
                family="dual_stationarity",
                sense=RowSense.EQUAL,
                classifiable=False,
            )
            for index in range(primal.n_columns)
        ),
        RowRecord(
            key="dual.near_optimal_face",
            family="dual_optimal_face",
            sense=RowSense.GREATER_EQUAL,
            scale=RowScale(),
            classifiable=False,
            metadata={"optimal_value": optimal_value, "delta": delta},
        ),
    )
    dual_model = CompiledLP(
        matrix,
        rhs,
        np.zeros(n_dual_columns, dtype=float),
        lower_bounds=np.asarray(lower, dtype=float),
        upper_bounds=np.asarray(upper, dtype=float),
        rows=rows,
        columns=tuple(columns),
    )
    return ExplicitDualLP(dual_model, inequality_columns)


def _status(status: ModelStatus) -> AuxiliaryStatus:
    if status is ModelStatus.OPTIMAL:
        return AuxiliaryStatus.OPTIMAL
    if status is ModelStatus.UNBOUNDED:
        return AuxiliaryStatus.UNBOUNDED
    if status in {ModelStatus.INFEASIBLE, ModelStatus.INFEASIBLE_OR_UNBOUNDED}:
        return AuxiliaryStatus.INFEASIBLE
    if status.is_limit:
        return AuxiliaryStatus.LIMIT
    return AuxiliaryStatus.FAILED


@dataclass(frozen=True)
class _RawRange:
    row_index: int
    lower: float | None
    upper: float | None
    lower_status: AuxiliaryStatus
    upper_status: AuxiliaryStatus
    lower_message: str
    upper_message: str


@dataclass(frozen=True)
class _DualWorkerContext:
    explicit: ExplicitDualLP
    report_factors: tuple[tuple[int, float], ...]
    tolerances: TolerancePolicy


_DUAL_WORKER_CONTEXT: _DualWorkerContext | None = None


def _initialize_dual_worker(
    explicit: ExplicitDualLP,
    report_factors: tuple[tuple[int, float], ...],
    tolerances: TolerancePolicy,
    expected_fingerprint: str,
) -> None:
    if explicit.model.fingerprint != expected_fingerprint:
        raise RuntimeError("explicit dual fingerprint changed at the process boundary")
    global _DUAL_WORKER_CONTEXT
    _DUAL_WORKER_CONTEXT = _DualWorkerContext(explicit, report_factors, tolerances)


def _solve_dual_worker(row_index: int) -> _RawRange:
    context = _DUAL_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("dual worker was not initialized")
    report_factor = dict(context.report_factors)[row_index]
    return _solve_range(context.explicit, row_index, report_factor, context.tolerances)


def _solve_range(
    explicit: ExplicitDualLP,
    row_index: int,
    report_factor: float,
    tolerances: TolerancePolicy,
) -> _RawRange:
    column = explicit.inequality_columns[row_index]
    unit = np.zeros(explicit.model.n_columns, dtype=float)
    unit[column] = 1.0 / report_factor
    values: list[float | None] = []
    statuses: list[AuxiliaryStatus] = []
    messages: list[str] = []
    try:
        with HighsBackend().session(
            explicit.model, solver_config(tolerances, solver="simplex")
        ) as session:
            basis = None
            for direction in ("min", "max"):
                objective = -unit if direction == "min" else unit
                session.set_objective(objective)
                result = session.solve(basis=basis)
                if result.basis is not None:
                    basis = result.basis
                status = _status(result.status)
                statuses.append(status)
                messages.append(result.message)
                certified, certificate_message = certified_optimal_result(result, tolerances)
                if certified and result.objective_value is not None:
                    value = float(result.objective_value)
                    values.append(-value if direction == "min" else value)
                elif direction == "max" and status is AuxiliaryStatus.UNBOUNDED:
                    values.append(float("inf"))
                else:
                    values.append(None)
                    if status is AuxiliaryStatus.OPTIMAL:
                        statuses[-1] = AuxiliaryStatus.FAILED
                        messages[-1] = certificate_message
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return _RawRange(
            row_index,
            None,
            None,
            AuxiliaryStatus.FAILED,
            AuxiliaryStatus.FAILED,
            message,
            message,
        )
    return _RawRange(
        row_index,
        values[0],
        values[1],
        statuses[0],
        statuses[1],
        messages[0],
        messages[1],
    )


def compute_dual_ranges(
    solution: DesignSolution,
    row_indices: tuple[int, ...],
    *,
    delta: float,
    n_jobs: int = 1,
    tolerances: TolerancePolicy | None = None,
) -> tuple[tuple[DualRangeResult, ...], tuple[TaskFailure, ...]]:
    """Compute reported-unit ranges through the explicit dual LP."""

    model = solution.compiled.model
    policy = tolerances or solution.tolerances
    for row_index in row_indices:
        if row_index < 0 or row_index >= model.n_rows:
            raise IndexError(f"dual range row index out of range: {row_index}")
        if model.rows[row_index].sense is not RowSense.GREATER_EQUAL:
            raise ValueError("dual activity ranges require inequality rows")
    if not row_indices:
        return (), ()
    try:
        explicit = build_explicit_dual(model, optimal_value=solution.objective_value, delta=delta)
    except Exception as exc:
        message = f"explicit dual construction failed: {type(exc).__name__}: {exc}"
        build_failures = tuple(
            TaskFailure(
                stage="dual_range_build",
                status=AuxiliaryStatus.FAILED,
                message=message,
                row_index=index,
                constraint_id=model.rows[index].key,
            )
            for index in row_indices
        )
        build_results = tuple(
            DualRangeResult(
                row_index=index,
                constraint_id=model.rows[index].key,
                family=model.rows[index].family,
                lower=None,
                upper=None,
                relevance=DualRelevance.INCONCLUSIVE,
                lower_status=AuxiliaryStatus.FAILED,
                upper_status=AuxiliaryStatus.FAILED,
                tolerance=policy.partition_dual,
                message=message,
            )
            for index in row_indices
        )
        return build_results, build_failures

    report_factors = tuple((index, model.rows[index].scale.report_factor) for index in row_indices)
    report_factor_by_row = dict(report_factors)
    workers = _resolve_workers(n_jobs, len(row_indices))
    if workers == 1:
        raw = tuple(
            _solve_range(explicit, index, report_factor_by_row[index], policy)
            for index in row_indices
        )
    else:
        ordered: list[_RawRange | None] = [None] * len(row_indices)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_context("spawn"),
                initializer=_initialize_dual_worker,
                initargs=(explicit, report_factors, policy, explicit.model.fingerprint),
            ) as pool:
                futures = {
                    pool.submit(_solve_dual_worker, index): position
                    for position, index in enumerate(row_indices)
                }
                for future in as_completed(futures):
                    position = futures[future]
                    try:
                        ordered[position] = future.result()
                    except Exception as exc:
                        index = row_indices[position]
                        message = f"{type(exc).__name__}: {exc}"
                        ordered[position] = _RawRange(
                            index,
                            None,
                            None,
                            AuxiliaryStatus.FAILED,
                            AuxiliaryStatus.FAILED,
                            message,
                            message,
                        )
        except Exception as exc:
            for position, item in enumerate(ordered):
                if item is None:
                    index = row_indices[position]
                    message = f"{type(exc).__name__}: {exc}"
                    ordered[position] = _RawRange(
                        index,
                        None,
                        None,
                        AuxiliaryStatus.FAILED,
                        AuxiliaryStatus.FAILED,
                        message,
                        message,
                    )
        raw = tuple(item for item in ordered if item is not None)

    tolerance = policy.partition_dual
    positive = 10.0 * max(tolerance, 1e-15)
    range_results: list[DualRangeResult] = []
    range_failures: list[TaskFailure] = []
    for item in raw:
        source = model.rows[item.row_index]
        lower = item.lower
        upper = item.upper
        relevance = DualRelevance.INCONCLUSIVE
        if lower is not None and upper is not None:
            if abs(lower) <= tolerance:
                lower = 0.0
            if np.isfinite(upper) and abs(upper) <= tolerance:
                upper = 0.0
            if upper <= positive:
                relevance = DualRelevance.NEVER_PRICED
            elif lower > positive:
                relevance = DualRelevance.UNIVERSALLY_PRICED
            elif lower <= tolerance and upper > positive:
                relevance = DualRelevance.POSSIBLY_PRICED
        messages = []
        if item.lower_status is not AuxiliaryStatus.OPTIMAL:
            messages.append(f"lower: {item.lower_message}")
            range_failures.append(
                TaskFailure(
                    stage="dual_range_lower",
                    status=item.lower_status,
                    message=item.lower_message,
                    row_index=item.row_index,
                    constraint_id=source.key,
                    direction="min",
                )
            )
        if item.upper_status not in {
            AuxiliaryStatus.OPTIMAL,
            AuxiliaryStatus.UNBOUNDED,
        }:
            messages.append(f"upper: {item.upper_message}")
            range_failures.append(
                TaskFailure(
                    stage="dual_range_upper",
                    status=item.upper_status,
                    message=item.upper_message,
                    row_index=item.row_index,
                    constraint_id=source.key,
                    direction="max",
                )
            )
        range_results.append(
            DualRangeResult(
                row_index=item.row_index,
                constraint_id=source.key,
                family=source.family,
                lower=lower,
                upper=upper,
                relevance=relevance,
                lower_status=item.lower_status,
                upper_status=item.upper_status,
                tolerance=tolerance,
                message="; ".join(messages) or None,
            )
        )
    return tuple(range_results), tuple(range_failures)
