"""Direct :mod:`highspy` backend for immutable MechAudit LPs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mechaudit.backends.base import (
    BackendFeatureError,
    BackendOperationError,
    BackendSession,
    BackendUnavailableError,
    BasisSnapshot,
    BoolArray,
    LPBackend,
    ModelStatus,
    SolverConfig,
    SolveResult,
)
from mechaudit.design.sparse_model import CompiledLP, SparseRow

IntArray: TypeAlias = NDArray[np.int32]


_REQUIRED_METHODS = (
    "clear",
    "getInfinity",
    "getObjectiveValue",
    "getOptionValue",
    "passModel",
    "run",
    "getModelStatus",
    "getSolution",
    "getInfo",
    "getBasis",
    "setBasis",
    "changeColsCost",
    "changeObjectiveOffset",
    "changeRowBounds",
    "addRows",
    "modelStatusToString",
    "setOptionValue",
    "version",
)

_CONTROLLED_OPTIONS = {
    "solver",
    "presolve",
    "run_crossover",
    "threads",
    "primal_feasibility_tolerance",
    "dual_feasibility_tolerance",
    "ipm_optimality_tolerance",
    "small_matrix_value",
    "time_limit",
    "random_seed",
    "output_flag",
}


def _load_highspy() -> Any:
    try:
        import highspy
    except ImportError as exc:  # pragma: no cover - dependency is required in normal installs
        raise BackendUnavailableError("the HiGHS backend requires the 'highspy' package") from exc
    return highspy


def _status_name(status: object) -> str:
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name
    text = str(status)
    return text.rsplit(".", 1)[-1]


def _check_ok(module: Any, status: object, operation: str) -> None:
    if status != module.HighsStatus.kOk:
        raise BackendOperationError(f"HiGHS {operation} returned {_status_name(status)}")


def _as_highs_indices(values: ArrayLike, *, name: str) -> IntArray:
    raw = np.asarray(values)
    if raw.size and (np.min(raw) < np.iinfo(np.int32).min or np.max(raw) > np.iinfo(np.int32).max):
        raise BackendFeatureError(f"{name} exceeds the 32-bit HiGHS index range")
    return np.ascontiguousarray(raw, dtype=np.int32)


def _as_highs_bounds(values: ArrayLike, infinity: float) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, order="C", copy=True)
    result[np.isposinf(result)] = infinity
    result[np.isneginf(result)] = -infinity
    return result


def _readonly_mask(values: ArrayLike) -> BoolArray:
    result = np.array(values, dtype=np.bool_, order="C", copy=True)
    result.flags.writeable = False
    return result


class HighsBackend(LPBackend):
    """Factory for one-shot and reusable direct-HiGHS LP solves."""

    name = "highs"

    def __init__(self) -> None:
        self._module = _load_highspy()
        self._check_features()

    @property
    def version(self) -> str:
        highs = self._module.Highs()
        return str(highs.version())

    @classmethod
    def available(cls) -> bool:
        """Return whether importable bindings expose the required API."""

        try:
            cls()
        except (BackendUnavailableError, BackendFeatureError):
            return False
        return True

    def solve(self, model: CompiledLP, config: SolverConfig | None = None) -> SolveResult:
        with self.session(model, config) as session:
            return session.solve()

    def session(self, model: CompiledLP, config: SolverConfig | None = None) -> HighsSession:
        if not isinstance(model, CompiledLP):
            raise TypeError("HighsBackend requires a CompiledLP")
        return HighsSession(self._module, model, config or SolverConfig())

    def _check_features(self) -> None:
        module = self._module
        missing_types = [
            name
            for name in (
                "Highs",
                "HighsLp",
                "HighsBasis",
                "HighsBasisStatus",
                "HighsStatus",
                "ObjSense",
                "MatrixFormat",
            )
            if not hasattr(module, name)
        ]
        if missing_types:
            raise BackendFeatureError(
                "installed highspy is missing: {}".format(", ".join(missing_types))
            )
        highs = module.Highs()
        missing_methods = [name for name in _REQUIRED_METHODS if not hasattr(highs, name)]
        if missing_methods:
            raise BackendFeatureError(
                "installed highspy lacks required methods: {}".format(", ".join(missing_methods))
            )
        if not hasattr(module.ObjSense, "kMaximize") or not hasattr(
            module.MatrixFormat, "kColwise"
        ):
            raise BackendFeatureError("installed highspy lacks canonical LP enum values")
        for option in sorted(_CONTROLLED_OPTIONS):
            status, _ = highs.getOptionValue(option)
            if status != module.HighsStatus.kOk:
                raise BackendFeatureError(
                    f"installed highspy does not expose required option {option!r}"
                )


class HighsSession(BackendSession):
    """A mutable HiGHS model supporting warm repeated auxiliary solves.

    Row removal is logical: the row bounds become ``[-inf, +inf]`` while its
    position and semantic record remain intact.  This makes it possible to
    evaluate the removed row's violation at the relaxed solution.
    """

    def __init__(self, module: Any, model: CompiledLP, config: SolverConfig) -> None:
        self._module = module
        self._model = model
        self._initial_objective = np.array(model.objective, dtype=np.float64, copy=True)
        self._initial_offset = model.objective_offset
        self._config = config
        self._highs = module.Highs()
        self._closed = False
        self._active_rows = np.ones(model.n_rows, dtype=np.bool_)
        self._infinity = float(self._highs.getInfinity())

        self._apply_options()
        data = model.solver_data()
        self._check_matrix_threshold(data.matrix.data)
        self._canonical_row_lower = _as_highs_bounds(data.row_lower, self._infinity)
        self._canonical_row_upper = _as_highs_bounds(data.row_upper, self._infinity)
        lp = self._make_highs_lp(data)
        _check_ok(module, self._highs.passModel(lp), "passModel")

    @property
    def model(self) -> CompiledLP:
        return self._model

    @property
    def active_rows(self) -> BoolArray:
        return _readonly_mask(self._active_rows)

    def solve(self, *, basis: BasisSnapshot | None = None) -> SolveResult:
        self._ensure_open()
        if basis is not None:
            self._set_basis(basis)

        run_status = self._highs.run()
        model_status_raw = self._highs.getModelStatus()
        status = _map_model_status(model_status_raw)
        warning = run_status == self._module.HighsStatus.kWarning
        if run_status == self._module.HighsStatus.kError:
            status = ModelStatus.ERROR

        message = str(self._highs.modelStatusToString(model_status_raw))
        if warning:
            message = f"{message} (HiGHS returned a warning status)"
        info = self._highs.getInfo()
        solution = self._highs.getSolution()
        primal_valid = bool(getattr(solution, "value_valid", False))
        dual_valid = bool(getattr(solution, "dual_valid", False))

        primal: NDArray[np.float64] | None = None
        residuals = None
        objective_value: float | None = None
        solver_objective: float | None = None
        if primal_valid:
            candidate = np.asarray(list(solution.col_value), dtype=np.float64)
            if candidate.size == self._model.n_columns and np.all(np.isfinite(candidate)):
                primal = candidate
                residuals = self._model.evaluate_primal(candidate, active_rows=self._active_rows)
                objective_value = self._model.objective_value(candidate)
                solver_objective = float(self._highs.getObjectiveValue())
            else:
                primal_valid = False

        solver_row_dual: NDArray[np.float64] | None = None
        solver_column_dual: NDArray[np.float64] | None = None
        reported_row_dual: NDArray[np.float64] | None = None
        if dual_valid:
            raw_rows = np.asarray(list(solution.row_dual), dtype=np.float64)
            raw_columns = np.asarray(list(solution.col_dual), dtype=np.float64)
            if (
                raw_rows.size == self._model.n_rows
                and raw_columns.size == self._model.n_columns
                and np.all(np.isfinite(raw_rows))
                and np.all(np.isfinite(raw_columns))
            ):
                solver_row_dual = raw_rows
                solver_column_dual = raw_columns
                # HiGHS row duals for a maximization lower-bound row are
                # nonpositive.  Negating produces the canonical nonnegative
                # multiplier for a >= row; RowScale then restores report units.
                reported_row_dual = np.array(
                    [
                        record.scale.solver_dual_to_reported(-raw_rows[index])
                        if self._active_rows[index]
                        else np.nan
                        for index, record in enumerate(self._model.rows)
                    ],
                    dtype=np.float64,
                )
            else:
                dual_valid = False

        basis_snapshot = self._get_basis()
        objective_bound = objective_value if status is ModelStatus.OPTIMAL else None
        solver_info = _extract_info(info)
        return SolveResult(
            status=status,
            solver_name="highs",
            solver_version=str(self._highs.version()),
            message=message,
            objective_value=objective_value,
            objective_bound=objective_bound,
            solver_objective_value=solver_objective,
            primal=primal,
            row_dual=reported_row_dual,
            solver_row_dual=solver_row_dual,
            solver_column_dual=solver_column_dual,
            residuals=residuals,
            active_rows=self._active_rows,
            basis=basis_snapshot,
            primal_solution_valid=primal_valid,
            dual_solution_valid=dual_valid,
            run_time_seconds=float(self._highs.getRunTime()),
            simplex_iterations=int(getattr(info, "simplex_iteration_count", 0)),
            ipm_iterations=int(getattr(info, "ipm_iteration_count", 0)),
            crossover_iterations=int(getattr(info, "crossover_iteration_count", 0)),
            solver_info=solver_info,
        )

    def set_objective(self, objective: ArrayLike, *, offset: float = 0.0) -> None:
        self._ensure_open()
        coefficients = np.array(objective, dtype=np.float64, order="C", copy=True)
        if coefficients.ndim != 1 or coefficients.size != self._model.n_columns:
            raise ValueError("objective length does not match the session model")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("objective coefficients must be finite")
        constant = float(offset)
        if not math.isfinite(constant):
            raise ValueError("objective offset must be finite")
        indices = np.arange(self._model.n_columns, dtype=np.int32)
        previous_objective = np.array(self._model.objective, dtype=np.float64, copy=True)
        previous_offset = self._model.objective_offset
        _check_ok(
            self._module,
            self._highs.changeColsCost(self._model.n_columns, indices, coefficients),
            "changeColsCost",
        )
        try:
            _check_ok(
                self._module,
                self._highs.changeObjectiveOffset(constant),
                "changeObjectiveOffset",
            )
        except BackendOperationError:
            # Preserve the session/model invariant if the second mutation fails.
            self._highs.changeColsCost(self._model.n_columns, indices, previous_objective)
            self._highs.changeObjectiveOffset(previous_offset)
            raise
        self._model = self._model.with_objective(coefficients, offset=constant)

    def reset_objective(self) -> None:
        self.set_objective(self._initial_objective, offset=self._initial_offset)

    def remove_rows(self, rows: Iterable[int | str]) -> tuple[int, ...]:
        indices = self._resolve_rows(rows)
        self._set_rows_active(indices, active=False)
        return indices

    def restore_rows(self, rows: Iterable[int | str]) -> tuple[int, ...]:
        indices = self._resolve_rows(rows)
        self._set_rows_active(indices, active=True)
        return indices

    def add_rows(self, rows: Sequence[SparseRow]) -> tuple[int, ...]:
        self._ensure_open()
        additions = tuple(rows)
        if not additions:
            return ()
        old_count = self._model.n_rows
        expanded = self._model.with_extra_rows(additions)
        data = expanded.solver_data()
        extra = data.matrix.tocsr()[old_count:, :]
        self._check_matrix_threshold(extra.data)
        starts = _as_highs_indices(extra.indptr[:-1], name="extra row starts")
        indices = _as_highs_indices(extra.indices, name="extra row column indices")
        values = np.ascontiguousarray(extra.data, dtype=np.float64)
        lower = _as_highs_bounds(data.row_lower[old_count:], self._infinity)
        upper = _as_highs_bounds(data.row_upper[old_count:], self._infinity)
        _check_ok(
            self._module,
            self._highs.addRows(len(additions), lower, upper, extra.nnz, starts, indices, values),
            "addRows",
        )
        self._model = expanded
        self._active_rows = np.concatenate(
            (self._active_rows, np.ones(len(additions), dtype=np.bool_))
        )
        self._canonical_row_lower = np.concatenate((self._canonical_row_lower, lower))
        self._canonical_row_upper = np.concatenate((self._canonical_row_upper, upper))
        return tuple(range(old_count, expanded.n_rows))

    def close(self) -> None:
        if not self._closed:
            self._highs.clear()
            self._closed = True

    def _apply_options(self) -> None:
        options: list[tuple[str, bool | int | float | str]] = [
            ("solver", self._config.solver),
            ("presolve", self._config.presolve),
            ("run_crossover", self._config.run_crossover),
            ("threads", self._config.threads),
            (
                "primal_feasibility_tolerance",
                self._config.primal_feasibility_tolerance,
            ),
            ("dual_feasibility_tolerance", self._config.dual_feasibility_tolerance),
            ("ipm_optimality_tolerance", self._config.ipm_optimality_tolerance),
            ("small_matrix_value", self._config.small_matrix_value),
            ("random_seed", self._config.random_seed),
            ("output_flag", self._config.output),
        ]
        if self._config.time_limit is not None:
            options.append(("time_limit", self._config.time_limit))
        for name, value in self._config.option_items:
            if name in _CONTROLLED_OPTIONS:
                raise BackendFeatureError(
                    f"extra_options cannot override controlled option {name!r}"
                )
            options.append((name, value))
        for name, value in options:
            _check_ok(
                self._module,
                self._highs.setOptionValue(name, value),
                f"setOptionValue({name!r})",
            )

    def _make_highs_lp(self, data: Any) -> Any:
        matrix = data.matrix
        if max(self._model.n_rows, self._model.n_columns, matrix.nnz) > np.iinfo(np.int32).max:
            raise BackendFeatureError("compiled LP exceeds the 32-bit HiGHS index range")
        lp = self._module.HighsLp()
        lp.num_col_ = self._model.n_columns
        lp.num_row_ = self._model.n_rows
        lp.sense_ = self._module.ObjSense.kMaximize
        lp.offset_ = self._model.objective_offset
        lp.col_cost_ = np.ascontiguousarray(data.objective, dtype=np.float64)
        lp.col_lower_ = _as_highs_bounds(data.lower_bounds, self._infinity)
        lp.col_upper_ = _as_highs_bounds(data.upper_bounds, self._infinity)
        lp.row_lower_ = self._canonical_row_lower.copy()
        lp.row_upper_ = self._canonical_row_upper.copy()
        lp.a_matrix_.format_ = self._module.MatrixFormat.kColwise
        lp.a_matrix_.start_ = _as_highs_indices(matrix.indptr, name="column starts")
        lp.a_matrix_.index_ = _as_highs_indices(matrix.indices, name="row indices")
        lp.a_matrix_.value_ = np.ascontiguousarray(matrix.data, dtype=np.float64)
        lp.col_names_ = [record.key for record in self._model.columns]
        lp.row_names_ = [record.key for record in self._model.rows]
        return lp

    def _check_matrix_threshold(self, coefficients: ArrayLike) -> None:
        values = np.abs(np.asarray(coefficients, dtype=np.float64))
        at_risk = values[(values > 0.0) & (values <= self._config.small_matrix_value)]
        if at_risk.size:
            raise BackendFeatureError(
                f"the scaled LP contains {at_risk.size} nonzero coefficient(s) at or below "
                f"HiGHS small_matrix_value={self._config.small_matrix_value:.3g}; "
                "rescale the affected rows or choose a supported smaller threshold"
            )

    def _resolve_rows(self, rows: Iterable[int | str]) -> tuple[int, ...]:
        self._ensure_open()
        resolved = []
        seen = set()
        for row in rows:
            index = self._model.row_index(row) if isinstance(row, str) else row
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("rows must be integer indices or stable keys")
            if index < 0 or index >= self._model.n_rows:
                raise IndexError(f"row index out of range: {index}")
            if index not in seen:
                resolved.append(index)
                seen.add(index)
        return tuple(resolved)

    def _set_rows_active(self, indices: tuple[int, ...], *, active: bool) -> None:
        self._ensure_open()
        changed = [index for index in indices if bool(self._active_rows[index]) != active]
        completed = []
        try:
            for index in changed:
                lower = self._canonical_row_lower[index] if active else -self._infinity
                upper = self._canonical_row_upper[index] if active else self._infinity
                _check_ok(
                    self._module,
                    self._highs.changeRowBounds(index, float(lower), float(upper)),
                    "changeRowBounds",
                )
                self._active_rows[index] = active
                completed.append(index)
        except BackendOperationError:
            # Best-effort rollback keeps session metadata aligned with HiGHS.
            for index in reversed(completed):
                rollback_active = not active
                lower = self._canonical_row_lower[index] if rollback_active else -self._infinity
                upper = self._canonical_row_upper[index] if rollback_active else self._infinity
                status = self._highs.changeRowBounds(index, float(lower), float(upper))
                if status == self._module.HighsStatus.kOk:
                    self._active_rows[index] = rollback_active
            raise

    def _get_basis(self) -> BasisSnapshot | None:
        basis = self._highs.getBasis()
        if not bool(getattr(basis, "valid", False)):
            return None
        columns = tuple(int(status.value) for status in basis.col_status)
        rows = tuple(int(status.value) for status in basis.row_status)
        if len(columns) != self._model.n_columns or len(rows) != self._model.n_rows:
            return None
        return BasisSnapshot(column_status=columns, row_status=rows, valid=True)

    def _set_basis(self, snapshot: BasisSnapshot) -> None:
        snapshot.validate_for(self._model)
        if not snapshot.valid:
            raise ValueError("cannot install an invalid basis snapshot")
        basis = self._module.HighsBasis()
        try:
            basis.col_status = [
                self._module.HighsBasisStatus(status) for status in snapshot.column_status
            ]
            basis.row_status = [
                self._module.HighsBasisStatus(status) for status in snapshot.row_status
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError("basis snapshot contains an unknown status code") from exc
        basis.valid = True
        _check_ok(self._module, self._highs.setBasis(basis), "setBasis")

    def _ensure_open(self) -> None:
        if self._closed:
            raise BackendOperationError("the HiGHS session is closed")


def _map_model_status(status: object) -> ModelStatus:
    mapping = {
        "kNotset": ModelStatus.NOT_SET,
        "kOptimal": ModelStatus.OPTIMAL,
        "kInfeasible": ModelStatus.INFEASIBLE,
        "kUnbounded": ModelStatus.UNBOUNDED,
        "kUnboundedOrInfeasible": ModelStatus.INFEASIBLE_OR_UNBOUNDED,
        "kTimeLimit": ModelStatus.TIME_LIMIT,
        "kIterationLimit": ModelStatus.ITERATION_LIMIT,
        "kObjectiveBound": ModelStatus.OBJECTIVE_LIMIT,
        "kObjectiveTarget": ModelStatus.OBJECTIVE_LIMIT,
        "kSolutionLimit": ModelStatus.SOLUTION_LIMIT,
        "kInterrupt": ModelStatus.INTERRUPTED,
        "kHighsInterrupt": ModelStatus.INTERRUPTED,
        "kMemoryLimit": ModelStatus.MEMORY_LIMIT,
        "kModelEmpty": ModelStatus.EMPTY,
        "kUnknown": ModelStatus.UNKNOWN,
        "kLoadError": ModelStatus.ERROR,
        "kModelError": ModelStatus.ERROR,
        "kPresolveError": ModelStatus.ERROR,
        "kSolveError": ModelStatus.ERROR,
        "kPostsolveError": ModelStatus.ERROR,
    }
    return mapping.get(_status_name(status), ModelStatus.UNKNOWN)


def _extract_info(info: object) -> tuple[tuple[str, int | float | str | bool], ...]:
    names = (
        "primal_solution_status",
        "dual_solution_status",
        "basis_validity",
        "max_primal_infeasibility",
        "max_dual_infeasibility",
        "max_primal_residual_error",
        "max_dual_residual_error",
        "max_complementarity_violation",
        "primal_dual_objective_error",
    )
    result = []
    for name in names:
        if not hasattr(info, name):
            continue
        value = getattr(info, name)
        if hasattr(value, "name"):
            value = str(value.name)
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (bool, int, float, str)):
            result.append((name, value))
    return tuple(result)
