"""Independent auxiliary LP tasks used by optimal-face diagnostics."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from mechaudit.backends.base import ModelStatus, SolverConfig, SolveResult
from mechaudit.backends.highs import HighsBackend
from mechaudit.design.linear import LinearExpression
from mechaudit.design.sparse_model import (
    CompiledLP,
    RowRecord,
    RowScale,
    RowSense,
    SparseRow,
)
from mechaudit.optimality.report import AuxiliaryStatus
from mechaudit.options import TolerancePolicy

FACE_ROW_PREFIX = "__mechaudit_optimal_face__"
_AUTO_WORKER_LIMIT = 4
_AUTO_PROCESS_MIN_TASKS = 512


def _row_expression(model: CompiledLP, row_index: int, *, factor: float = 1.0) -> LinearExpression:
    row = model.matrix.getrow(row_index)
    return LinearExpression(
        tuple(
            (int(column), factor * float(coefficient))
            for column, coefficient in zip(row.indices, row.data, strict=True)
        )
    )


def make_face_model(
    model: CompiledLP,
    *,
    optimal_value: float,
    delta: float,
    key_prefix: str = FACE_ROW_PREFIX,
) -> CompiledLP:
    """Append ``c @ z + offset >= optimal_value - delta`` in original units."""

    if delta < 0.0 or not np.isfinite(delta):
        raise ValueError("optimal-face delta must be finite and nonnegative")
    existing = {row.key for row in model.rows}
    key = key_prefix
    suffix = 1
    while key in existing:
        key = f"{key_prefix}[{suffix}]"
        suffix += 1
    expression = LinearExpression(
        tuple((index, float(value)) for index, value in enumerate(model.objective) if value != 0.0),
        constant=model.objective_offset,
    )
    return model.with_extra_rows(
        (
            SparseRow(
                expression=expression,
                rhs=float(optimal_value) - float(delta),
                record=RowRecord(
                    key=key,
                    family="optimal_face",
                    sense=RowSense.GREATER_EQUAL,
                    scale=RowScale(),
                    classifiable=False,
                    name="near-optimal objective face",
                    metadata={"optimal_value": optimal_value, "delta": delta},
                ),
            ),
        )
    )


def solver_config(
    tolerances: TolerancePolicy,
    *,
    solver: str = "simplex",
    presolve: str = "choose",
    crossover: str = "choose",
) -> SolverConfig:
    """Translate semantic tolerances to supported HiGHS tolerances."""

    floor = 1e-10
    primal = max(floor, float(tolerances.feasibility))
    dual = max(floor, float(tolerances.partition_dual))
    ipm = max(floor, min(primal, dual, max(floor, tolerances.complementarity)))
    return SolverConfig(
        solver=solver,
        presolve=presolve,
        run_crossover=crossover,
        threads=1,
        primal_feasibility_tolerance=primal,
        dual_feasibility_tolerance=dual,
        ipm_optimality_tolerance=ipm,
        small_matrix_value=1e-12,
        random_seed=0,
        output=False,
    )


def certified_optimal_result(result: SolveResult, tolerances: TolerancePolicy) -> tuple[bool, str]:
    """Check the solver optimum, bound, and independent primal/dual residuals."""

    if result.status is not ModelStatus.OPTIMAL:
        return False, f"solver status is {result.status.value}, not optimal"
    if not result.has_primal_solution or not result.has_dual_solution:
        return False, "optimal status lacks a valid primal-dual solution"
    if result.residuals is None or result.residuals.max_violation > tolerances.feasibility:
        return False, "independent primal residual exceeds the feasibility tolerance"
    if result.objective_value is None or result.objective_bound is None:
        return False, "optimal result has no objective value/bound"
    margin = max(tolerances.objective_absolute, tolerances.feasibility, 1e-10)
    if abs(result.objective_bound - result.objective_value) > margin * max(
        1.0, abs(result.objective_value)
    ):
        return False, "reported objective value and objective bound are inconsistent"
    info = dict(result.solver_info)
    dual_infeasibility = info.get("max_dual_infeasibility")
    if (
        not isinstance(dual_infeasibility, (int, float))
        or not np.isfinite(float(dual_infeasibility))
        or float(dual_infeasibility) > max(tolerances.partition_dual, 1e-10)
    ):
        return False, "dual infeasibility exceeds the partition tolerance"
    objective_error = info.get("primal_dual_objective_error")
    if (
        not isinstance(objective_error, (int, float))
        or not np.isfinite(float(objective_error))
        or float(objective_error)
        > max(tolerances.complementarity, tolerances.objective_absolute, 1e-10)
    ):
        return False, "primal-dual objective error exceeds tolerance"
    return True, "certified optimal solver result"


@dataclass(frozen=True)
class SlackTask:
    """Maximize or minimize one reported-unit row slack."""

    row_index: int
    direction: str
    removed_rows: tuple[int, ...] = ()
    label: str = "slack_range"

    def __post_init__(self) -> None:
        if self.direction not in {"max", "min"}:
            raise ValueError("slack task direction must be 'max' or 'min'")


@dataclass(frozen=True)
class SlackTaskResult:
    """Serializable result of an independent slack optimization."""

    task: SlackTask
    status: AuxiliaryStatus
    solver_status: str
    message: str
    slack: float | None = None
    design_objective_value: float | None = None
    primal: tuple[float, ...] | None = None
    max_primal_violation: float | None = None
    auxiliary_objective_value: float | None = None
    objective_bound: float | None = None
    max_dual_infeasibility: float | None = None
    primal_dual_objective_error: float | None = None
    run_time_seconds: float | None = None

    @property
    def has_witness(self) -> bool:
        return self.primal is not None


@dataclass(frozen=True)
class _SlackWorkerContext:
    base_model: CompiledLP
    face_model: CompiledLP
    optimal_value: float
    delta: float
    tolerances: TolerancePolicy
    require_face: bool


_SLACK_WORKER_CONTEXT: _SlackWorkerContext | None = None


def _process_entry_point_available() -> bool:
    """Return whether spawn workers can safely re-import the main program."""

    main = sys.modules.get("__main__")
    raw_path = getattr(main, "__file__", None)
    return isinstance(raw_path, str) and not raw_path.startswith("<") and Path(raw_path).exists()


def _resolve_workers(n_jobs: int, task_count: int) -> int:
    """Resolve explicit/automatic process counts without oversubscribing by default."""

    if task_count < 0:
        raise ValueError("task_count must be nonnegative")
    if task_count == 0:
        return 0
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    if n_jobs == -1:
        workers = min(os.cpu_count() or 1, _AUTO_WORKER_LIMIT, task_count)
        if task_count < max(_AUTO_PROCESS_MIN_TASKS, 2 * workers):
            return 1
    else:
        workers = min(n_jobs, task_count)
    if workers > 1 and not _process_entry_point_available():
        return 1
    return max(1, int(workers))


def _initialize_slack_worker(
    base_model: CompiledLP,
    face_model: CompiledLP,
    optimal_value: float,
    delta: float,
    tolerances: TolerancePolicy,
    require_face: bool,
    expected_base_fingerprint: str,
    expected_face_fingerprint: str,
) -> None:
    """Install an integrity-checked immutable context once per worker."""

    if base_model.fingerprint != expected_base_fingerprint:
        raise RuntimeError("base model fingerprint changed at the process boundary")
    if face_model.fingerprint != expected_face_fingerprint:
        raise RuntimeError("face model fingerprint changed at the process boundary")
    global _SLACK_WORKER_CONTEXT
    _SLACK_WORKER_CONTEXT = _SlackWorkerContext(
        base_model,
        face_model,
        optimal_value,
        delta,
        tolerances,
        require_face,
    )


def _solve_slack_worker(task: SlackTask) -> SlackTaskResult:
    context = _SLACK_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("slack worker was not initialized")
    return solve_slack_task(
        context.base_model,
        context.face_model,
        task,
        optimal_value=context.optimal_value,
        delta=context.delta,
        tolerances=context.tolerances,
        require_face=context.require_face,
    )


def _auxiliary_status(status: ModelStatus) -> AuxiliaryStatus:
    if status is ModelStatus.OPTIMAL:
        return AuxiliaryStatus.OPTIMAL
    if status is ModelStatus.UNBOUNDED:
        return AuxiliaryStatus.UNBOUNDED
    if status in {ModelStatus.INFEASIBLE, ModelStatus.INFEASIBLE_OR_UNBOUNDED}:
        return AuxiliaryStatus.INFEASIBLE
    if status.is_limit:
        return AuxiliaryStatus.LIMIT
    return AuxiliaryStatus.FAILED


def solve_slack_task(
    base_model: CompiledLP,
    face_model: CompiledLP,
    task: SlackTask,
    *,
    optimal_value: float,
    delta: float,
    tolerances: TolerancePolicy,
    require_face: bool = True,
) -> SlackTaskResult:
    """Execute one task in a fresh one-thread HiGHS instance."""

    if task.row_index < 0 or task.row_index >= base_model.n_rows:
        raise IndexError("slack task row index is outside the base model")
    record = base_model.rows[task.row_index]
    sign = 1.0 if task.direction == "max" else -1.0
    report_factor = record.scale.report_factor
    objective = np.zeros(base_model.n_columns, dtype=float)
    row = base_model.matrix.getrow(task.row_index)
    objective[row.indices] = sign * report_factor * row.data
    offset = -sign * report_factor * float(base_model.rhs[task.row_index])

    try:
        backend = HighsBackend()
        with backend.session(face_model, solver_config(tolerances)) as session:
            if task.removed_rows:
                session.remove_rows(task.removed_rows)
            session.set_objective(objective, offset=offset)
            result = session.solve()
    except Exception as exc:  # backend failures are retained, never dropped
        return SlackTaskResult(
            task=task,
            status=AuxiliaryStatus.FAILED,
            solver_status="exception",
            message=f"{type(exc).__name__}: {exc}",
        )

    status = _auxiliary_status(result.status)
    result_info = dict(result.solver_info)
    raw_dual_infeasibility = result_info.get("max_dual_infeasibility")
    raw_objective_error = result_info.get("primal_dual_objective_error")
    dual_infeasibility = (
        float(raw_dual_infeasibility) if isinstance(raw_dual_infeasibility, (int, float)) else None
    )
    objective_error = (
        float(raw_objective_error) if isinstance(raw_objective_error, (int, float)) else None
    )
    if status is not AuxiliaryStatus.OPTIMAL:
        return SlackTaskResult(
            task=task,
            status=status,
            solver_status=result.status.value,
            message=result.message,
            run_time_seconds=result.run_time_seconds,
        )
    if not result.has_primal_solution or result.primal is None:
        return SlackTaskResult(
            task=task,
            status=AuxiliaryStatus.FAILED,
            solver_status=result.status.value,
            message="solver reported optimality without a valid primal solution",
            run_time_seconds=result.run_time_seconds,
        )

    primal = result.primal
    residuals = base_model.evaluate_primal(primal)
    original_slack = float(residuals.row_slack[task.row_index])
    reported_slack = record.scale.original_slack_to_reported(original_slack)
    design_value = base_model.objective_value(primal)
    max_violation = None if result.residuals is None else float(result.residuals.max_violation)
    numerical_margin = max(tolerances.feasibility, tolerances.objective_absolute)
    expected_target = reported_slack if task.direction == "max" else -reported_slack
    objective_consistent = (
        result.objective_value is not None
        and result.objective_bound is not None
        and abs(result.objective_value - expected_target)
        <= numerical_margin * max(1.0, abs(expected_target))
        and abs(result.objective_bound - result.objective_value)
        <= numerical_margin * max(1.0, abs(result.objective_value))
    )
    certified, certificate_message = certified_optimal_result(result, tolerances)
    if max_violation is None or max_violation > tolerances.feasibility:
        return SlackTaskResult(
            task=task,
            status=AuxiliaryStatus.FAILED,
            solver_status=result.status.value,
            message="independent primal residual exceeds the feasibility tolerance",
            slack=reported_slack,
            design_objective_value=design_value,
            primal=tuple(float(value) for value in primal),
            max_primal_violation=max_violation,
            auxiliary_objective_value=result.objective_value,
            objective_bound=result.objective_bound,
            max_dual_infeasibility=dual_infeasibility,
            primal_dual_objective_error=objective_error,
            run_time_seconds=result.run_time_seconds,
        )
    if not objective_consistent or not certified:
        return SlackTaskResult(
            task=task,
            status=AuxiliaryStatus.FAILED,
            solver_status=result.status.value,
            message=(f"solver optimum lacks a consistent target objective; {certificate_message}"),
            slack=reported_slack,
            design_objective_value=design_value,
            primal=tuple(float(value) for value in primal),
            max_primal_violation=max_violation,
            auxiliary_objective_value=result.objective_value,
            objective_bound=result.objective_bound,
            max_dual_infeasibility=dual_infeasibility,
            primal_dual_objective_error=objective_error,
            run_time_seconds=result.run_time_seconds,
        )
    if require_face and design_value < optimal_value - delta - numerical_margin:
        return SlackTaskResult(
            task=task,
            status=AuxiliaryStatus.FAILED,
            solver_status=result.status.value,
            message="auxiliary witness lies outside the declared near-optimal face",
            slack=reported_slack,
            design_objective_value=design_value,
            primal=tuple(float(value) for value in primal),
            max_primal_violation=max_violation,
            auxiliary_objective_value=result.objective_value,
            objective_bound=result.objective_bound,
            max_dual_infeasibility=dual_infeasibility,
            primal_dual_objective_error=objective_error,
            run_time_seconds=result.run_time_seconds,
        )
    return SlackTaskResult(
        task=task,
        status=AuxiliaryStatus.OPTIMAL,
        solver_status=result.status.value,
        message=result.message,
        slack=reported_slack,
        design_objective_value=design_value,
        primal=tuple(float(value) for value in primal),
        max_primal_violation=max_violation,
        auxiliary_objective_value=result.objective_value,
        objective_bound=result.objective_bound,
        max_dual_infeasibility=dual_infeasibility,
        primal_dual_objective_error=objective_error,
        run_time_seconds=result.run_time_seconds,
    )


def run_slack_tasks(
    base_model: CompiledLP,
    face_model: CompiledLP,
    tasks: tuple[SlackTask, ...],
    *,
    optimal_value: float,
    delta: float,
    tolerances: TolerancePolicy,
    n_jobs: int = 1,
    require_face: bool = True,
) -> tuple[SlackTaskResult, ...]:
    """Run independent tasks deterministically, with one solver thread per process."""

    if not tasks:
        return ()
    workers = _resolve_workers(n_jobs, len(tasks))
    if workers == 1:
        return tuple(
            solve_slack_task(
                base_model,
                face_model,
                task,
                optimal_value=optimal_value,
                delta=delta,
                tolerances=tolerances,
                require_face=require_face,
            )
            for task in tasks
        )

    results: list[SlackTaskResult | None] = [None] * len(tasks)
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_slack_worker,
            initargs=(
                base_model,
                face_model,
                optimal_value,
                delta,
                tolerances,
                require_face,
                base_model.fingerprint,
                face_model.fingerprint,
            ),
        ) as pool:
            futures = {
                pool.submit(_solve_slack_worker, task): index for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:  # worker failures remain visible and ordered
                    results[index] = SlackTaskResult(
                        task=tasks[index],
                        status=AuxiliaryStatus.FAILED,
                        solver_status="exception",
                        message=f"{type(exc).__name__}: {exc}",
                    )
    except Exception as exc:
        for index, result in enumerate(results):
            if result is None:
                results[index] = SlackTaskResult(
                    task=tasks[index],
                    status=AuxiliaryStatus.FAILED,
                    solver_status="process_pool_exception",
                    message=f"{type(exc).__name__}: {exc}",
                )
    return tuple(result for result in results if result is not None)
