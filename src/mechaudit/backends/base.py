"""Solver-independent backend contracts and result objects."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mechaudit.design.sparse_model import CompiledLP, PrimalResiduals, SparseRow

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
OptionValue: TypeAlias = bool | int | float | str


class BackendError(RuntimeError):
    """Base class for backend availability, feature, and execution errors."""


class BackendUnavailableError(BackendError):
    """The requested solver Python bindings are not importable."""


class BackendFeatureError(BackendError):
    """The installed backend lacks a required low-level feature."""


class BackendOperationError(BackendError):
    """A low-level backend operation returned an error status."""


class ModelStatus(str, Enum):
    """Normalized LP termination states."""

    NOT_SET = "not_set"
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    INFEASIBLE_OR_UNBOUNDED = "infeasible_or_unbounded"
    TIME_LIMIT = "time_limit"
    ITERATION_LIMIT = "iteration_limit"
    OBJECTIVE_LIMIT = "objective_limit"
    SOLUTION_LIMIT = "solution_limit"
    INTERRUPTED = "interrupted"
    MEMORY_LIMIT = "memory_limit"
    EMPTY = "empty"
    UNKNOWN = "unknown"
    ERROR = "error"

    @property
    def is_limit(self) -> bool:
        return self in {
            ModelStatus.TIME_LIMIT,
            ModelStatus.ITERATION_LIMIT,
            ModelStatus.OBJECTIVE_LIMIT,
            ModelStatus.SOLUTION_LIMIT,
            ModelStatus.INTERRUPTED,
            ModelStatus.MEMORY_LIMIT,
        }


@dataclass(frozen=True)
class SolverConfig:
    """Portable LP solve settings used by all version-0.1 backends."""

    solver: str = "simplex"
    presolve: str = "choose"
    run_crossover: str = "choose"
    threads: int = 1
    primal_feasibility_tolerance: float = 1e-9
    dual_feasibility_tolerance: float = 1e-9
    ipm_optimality_tolerance: float = 1e-9
    small_matrix_value: float = 1e-12
    time_limit: float | None = None
    random_seed: int = 0
    output: bool = False
    extra_options: Mapping[str, OptionValue] | Iterable[tuple[str, OptionValue]] | None = ()

    def __post_init__(self) -> None:
        if self.solver not in {"choose", "simplex", "ipm"}:
            raise ValueError("solver must be 'choose', 'simplex', or 'ipm'")
        if self.presolve not in {"off", "choose", "on"}:
            raise ValueError("presolve must be 'off', 'choose', or 'on'")
        if self.run_crossover not in {"off", "choose", "on"}:
            raise ValueError("run_crossover must be 'off', 'choose', or 'on'")
        if isinstance(self.threads, bool) or not isinstance(self.threads, int):
            raise TypeError("threads must be an integer")
        if self.threads < 1:
            raise ValueError("threads must be at least one")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer")
        if self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative")

        for name in (
            "primal_feasibility_tolerance",
            "dual_feasibility_tolerance",
            "ipm_optimality_tolerance",
        ):
            tolerance_value = float(getattr(self, name))
            if not math.isfinite(tolerance_value) or tolerance_value < 1e-10:
                raise ValueError(f"{name} must be finite and at least 1e-10")
            object.__setattr__(self, name, tolerance_value)
        small = float(self.small_matrix_value)
        if not math.isfinite(small) or small < 1e-12:
            raise ValueError("small_matrix_value must be finite and at least 1e-12")
        object.__setattr__(self, "small_matrix_value", small)
        if self.time_limit is not None:
            limit = float(self.time_limit)
            if not math.isfinite(limit) or limit < 0.0:
                raise ValueError("time_limit must be finite and nonnegative")
            object.__setattr__(self, "time_limit", limit)

        raw_options = self.extra_options
        if isinstance(raw_options, Mapping):
            items = tuple(raw_options.items())
        elif raw_options is None:
            items = ()
        else:
            items = tuple(raw_options)
        canonical: list[tuple[str, OptionValue]] = []
        for key, option_value in items:
            if not isinstance(key, str) or not key:
                raise TypeError("extra solver option names must be non-empty strings")
            raw_value: object = option_value
            if isinstance(raw_value, bool):
                canonical_value: OptionValue = raw_value
            elif isinstance(raw_value, int):
                canonical_value = raw_value
            elif isinstance(raw_value, float):
                if not math.isfinite(raw_value):
                    raise ValueError("extra floating solver options must be finite")
                canonical_value = raw_value
            elif isinstance(raw_value, str):
                canonical_value = raw_value
            else:
                raise TypeError(
                    f"unsupported value for solver option {key!r}: {type(raw_value).__name__}"
                )
            canonical.append((key, canonical_value))
        canonical.sort(key=lambda item: item[0])
        if len({key for key, _ in canonical}) != len(canonical):
            raise ValueError("duplicate extra solver option")
        object.__setattr__(self, "extra_options", tuple(canonical))

    @property
    def option_items(self) -> tuple[tuple[str, OptionValue], ...]:
        return cast(tuple[tuple[str, OptionValue], ...], self.extra_options)

    def with_overrides(self, **changes: Any) -> SolverConfig:
        """Return an immutable configuration copy with selected replacements."""

        return replace(self, **changes)


@dataclass(frozen=True)
class BasisSnapshot:
    """Solver-independent serialization of a simplex basis."""

    column_status: tuple[int, ...]
    row_status: tuple[int, ...]
    valid: bool = True

    def validate_for(self, model: CompiledLP) -> None:
        if len(self.column_status) != model.n_columns:
            raise ValueError("basis column count does not match the model")
        if len(self.row_status) != model.n_rows:
            raise ValueError("basis row count does not match the model")


def _optional_float_array(
    values: ArrayLike | None, *, length: int | None = None
) -> FloatArray | None:
    if values is None:
        return None
    array = np.array(values, dtype=np.float64, order="C", copy=True)
    if array.ndim != 1:
        raise ValueError("solver result arrays must be one-dimensional")
    if length is not None and array.size != length:
        raise ValueError("solver result array length does not match the model")
    array[array == 0.0] = 0.0
    array.flags.writeable = False
    return array


def _optional_bool_array(
    values: ArrayLike | None, *, length: int | None = None
) -> BoolArray | None:
    if values is None:
        return None
    array = np.array(values, dtype=np.bool_, order="C", copy=True)
    if array.ndim != 1:
        raise ValueError("solver result masks must be one-dimensional")
    if length is not None and array.size != length:
        raise ValueError("solver result mask length does not match the model")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class SolveResult:
    """Normalized solve result with independently recomputed primal residuals."""

    status: ModelStatus
    solver_name: str
    solver_version: str
    message: str
    objective_value: float | None = None
    objective_bound: float | None = None
    solver_objective_value: float | None = None
    primal: FloatArray | None = None
    row_dual: FloatArray | None = None
    solver_row_dual: FloatArray | None = None
    solver_column_dual: FloatArray | None = None
    residuals: PrimalResiduals | None = None
    active_rows: BoolArray | None = None
    basis: BasisSnapshot | None = None
    primal_solution_valid: bool = False
    dual_solution_valid: bool = False
    run_time_seconds: float | None = None
    simplex_iterations: int = 0
    ipm_iterations: int = 0
    crossover_iterations: int = 0
    solver_info: tuple[tuple[str, int | float | str | bool], ...] = ()

    def __post_init__(self) -> None:
        primal = _optional_float_array(self.primal)
        row_dual = _optional_float_array(self.row_dual)
        solver_row_dual = _optional_float_array(self.solver_row_dual)
        solver_column_dual = _optional_float_array(self.solver_column_dual)
        active_rows = _optional_bool_array(self.active_rows)
        object.__setattr__(self, "primal", primal)
        object.__setattr__(self, "row_dual", row_dual)
        object.__setattr__(self, "solver_row_dual", solver_row_dual)
        object.__setattr__(self, "solver_column_dual", solver_column_dual)
        object.__setattr__(self, "active_rows", active_rows)
        for name in ("objective_value", "objective_bound", "solver_objective_value"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))
        if self.run_time_seconds is not None:
            object.__setattr__(self, "run_time_seconds", float(self.run_time_seconds))
        object.__setattr__(self, "solver_info", tuple(self.solver_info))

    @property
    def is_optimal(self) -> bool:
        return self.status is ModelStatus.OPTIMAL

    @property
    def has_primal_solution(self) -> bool:
        return self.primal_solution_valid and self.primal is not None

    @property
    def has_dual_solution(self) -> bool:
        return self.dual_solution_valid and self.row_dual is not None


class LPBackend(ABC):
    """Factory for one-shot solves and reusable mutable sessions."""

    name: str

    @abstractmethod
    def solve(self, model: CompiledLP, config: SolverConfig | None = None) -> SolveResult:
        """Solve one immutable compiled LP."""

    @abstractmethod
    def session(self, model: CompiledLP, config: SolverConfig | None = None) -> BackendSession:
        """Create a reusable session for auxiliary objectives and row changes."""


class BackendSession(ABC):
    """Mutable solver state wrapped around immutable model snapshots."""

    @property
    @abstractmethod
    def model(self) -> CompiledLP:
        """Current immutable model snapshot, including appended rows/objective."""

    @property
    @abstractmethod
    def active_rows(self) -> BoolArray:
        """Mask of rows currently enforced by the backend."""

    @abstractmethod
    def solve(self, *, basis: BasisSnapshot | None = None) -> SolveResult:
        """Solve the current session model."""

    @abstractmethod
    def set_objective(self, objective: ArrayLike, *, offset: float = 0.0) -> None:
        """Replace the maximization objective in place."""

    @abstractmethod
    def reset_objective(self) -> None:
        """Restore the session's initially compiled objective."""

    @abstractmethod
    def remove_rows(self, rows: Iterable[int | str]) -> tuple[int, ...]:
        """Logically remove rows while preserving their stable indices."""

    @abstractmethod
    def restore_rows(self, rows: Iterable[int | str]) -> tuple[int, ...]:
        """Restore rows to their canonical bounds."""

    @abstractmethod
    def add_rows(self, rows: Sequence[SparseRow]) -> tuple[int, ...]:
        """Append rows and return their stable session indices."""

    @abstractmethod
    def close(self) -> None:
        """Release backend-owned model and solver state."""

    @contextmanager
    def rows_removed(self, rows: Iterable[int | str]) -> Iterator[tuple[int, ...]]:
        """Temporarily disable rows and restore them even after an exception."""

        indices = self.remove_rows(rows)
        try:
            yield indices
        finally:
            self.restore_rows(indices)

    def __enter__(self) -> BackendSession:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()
