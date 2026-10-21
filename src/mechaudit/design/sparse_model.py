"""Immutable, solver-independent sparse linear-program representation.

MechAudit compiles every design problem to the following canonical form::

    maximize    c @ z + offset
    subject to  A[j] @ z >= rhs[j]   (inequality rows)
                A[j] @ z == rhs[j]   (equality rows)
                lower <= z <= upper

``CompiledLP`` stores ``A`` and all reported quantities in original economic
units.  Positive row scales are metadata for the backend view only.  This
separation is important: solver scaling must never change the slack or dual
units shown to researchers.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from mechaudit.design.linear import LinearExpression

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


class _HashLike(Protocol):
    def update(self, data: bytes) -> None: ...


def _readonly_float_array(
    values: ArrayLike, *, name: str, length: int | None = None, finite: bool = True
) -> FloatArray:
    array = np.array(values, dtype=np.float64, order="C", copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and array.size != length:
        raise ValueError(f"{name} has length {array.size}; expected {length}")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if not finite and np.any(np.isnan(array)):
        raise ValueError(f"{name} cannot contain NaN")
    array[array == 0.0] = 0.0  # Canonicalize negative zero.
    array.flags.writeable = False
    return array


def _readonly_bool_array(values: ArrayLike, *, name: str, length: int | None = None) -> BoolArray:
    array = np.array(values, dtype=np.bool_, order="C", copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and array.size != length:
        raise ValueError(f"{name} has length {array.size}; expected {length}")
    array.flags.writeable = False
    return array


def _view(array: FloatArray) -> FloatArray:
    result = array.view()
    result.flags.writeable = False
    return result


def _bool_view(array: BoolArray) -> BoolArray:
    result = array.view()
    result.flags.writeable = False
    return result


def _freeze_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("row/column metadata cannot contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        frozen = [
            (_freeze_metadata_value(key), _freeze_metadata_value(item))
            for key, item in value.items()
        ]
        frozen.sort(key=lambda pair: (type(pair[0]).__name__, repr(pair[0])))
        return tuple(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata_value(item) for item in value)
    raise TypeError(
        f"row/column metadata values must be immutable JSON-like values; got {type(value).__name__}"
    )


def _freeze_metadata(
    metadata: Mapping[str, Any] | Iterable[tuple[str, Any]] | None,
) -> tuple[tuple[str, Any], ...]:
    if metadata is None:
        return ()
    items = metadata.items() if isinstance(metadata, Mapping) else metadata
    frozen = []
    for key, value in items:
        if not isinstance(key, str) or not key:
            raise TypeError("row/column metadata keys must be non-empty strings")
        frozen.append((key, _freeze_metadata_value(value)))
    frozen.sort(key=lambda item: item[0])
    if len({key for key, _ in frozen}) != len(frozen):
        raise ValueError("row/column metadata contains duplicate keys")
    return tuple(frozen)


class RowSense(str, Enum):
    """Canonical row senses supported by :class:`CompiledLP`."""

    GREATER_EQUAL = ">="
    EQUAL = "=="


@dataclass(frozen=True)
class RowScale:
    """Positive transformations between original, solver, and report units.

    If ``factor`` is :math:`\alpha`, the solver receives
    ``alpha * (A[j] @ z - rhs[j])``.  If ``report_factor`` is :math:`\beta`,
    the reported slack is ``beta`` times the original slack.  Consequently a
    normalized solver multiplier maps to report units as
    ``factor / report_factor`` times that multiplier.
    """

    factor: float = 1.0
    report_factor: float = 1.0
    original_unit: str | None = None
    report_unit: str | None = None

    def __post_init__(self) -> None:
        factor = float(self.factor)
        report_factor = float(self.report_factor)
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("solver row scale must be finite and strictly positive")
        if not math.isfinite(report_factor) or report_factor <= 0.0:
            raise ValueError("report row scale must be finite and strictly positive")
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "report_factor", report_factor)

    def original_slack_to_solver(self, slack: float) -> float:
        return self.factor * float(slack)

    def solver_slack_to_original(self, slack: float) -> float:
        return float(slack) / self.factor

    def original_slack_to_reported(self, slack: float) -> float:
        return self.report_factor * float(slack)

    def solver_dual_to_reported(self, multiplier: float) -> float:
        return (self.factor / self.report_factor) * float(multiplier)


@dataclass(frozen=True)
class RowRecord:
    """Stable semantic metadata for one compiled row."""

    key: str
    family: str
    sense: RowSense = RowSense.GREATER_EQUAL
    scale: RowScale = RowScale()
    classifiable: bool = True
    name: str | None = None
    metadata: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("row key must be a non-empty stable string")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("row family must be a non-empty string")
        try:
            sense = self.sense if isinstance(self.sense, RowSense) else RowSense(self.sense)
        except ValueError as exc:
            raise ValueError(f"unsupported canonical row sense: {self.sense!r}") from exc
        if sense is RowSense.EQUAL and self.classifiable:
            object.__setattr__(self, "classifiable", False)
        object.__setattr__(self, "sense", sense)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def metadata_dict(self) -> Mapping[str, Any]:
        """Return a detached read-only-style metadata mapping."""

        return dict(cast(Iterable[tuple[str, Any]], self.metadata or ()))


@dataclass(frozen=True)
class ColumnRecord:
    """Stable metadata for one compiled LP column."""

    key: str
    block: str
    name: str | None = None
    metadata: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("column key must be a non-empty stable string")
        if not isinstance(self.block, str) or not self.block:
            raise ValueError("column block must be a non-empty string")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def metadata_dict(self) -> Mapping[str, Any]:
        return dict(cast(Iterable[tuple[str, Any]], self.metadata or ()))


@dataclass(frozen=True)
class SparseRow:
    """A row that can be appended to an existing :class:`CompiledLP`."""

    expression: LinearExpression
    rhs: float
    record: RowRecord

    def __post_init__(self) -> None:
        if not isinstance(self.expression, LinearExpression):
            raise TypeError("SparseRow.expression must be a LinearExpression")
        rhs = float(self.rhs)
        if not math.isfinite(rhs):
            raise ValueError("sparse-row right-hand side must be finite")
        object.__setattr__(self, "rhs", rhs)

    @property
    def canonical_rhs(self) -> float:
        """RHS after moving the expression constant to the right."""

        return self.rhs - self.expression.constant


@dataclass(frozen=True)
class SolverLPData:
    """Scaled CSC data ready for a low-level LP backend."""

    matrix: sparse.csc_matrix
    objective: FloatArray
    objective_offset: float
    lower_bounds: FloatArray
    upper_bounds: FloatArray
    row_lower: FloatArray
    row_upper: FloatArray


@dataclass(frozen=True)
class PrimalResiduals:
    """Independent primal residuals in the model's original units."""

    row_activity: FloatArray
    row_slack: FloatArray
    row_violation: FloatArray
    lower_bound_violation: FloatArray
    upper_bound_violation: FloatArray
    active_rows: BoolArray
    max_row_violation: float
    max_bound_violation: float
    max_violation: float

    def is_feasible(self, tolerance: float) -> bool:
        if tolerance < 0.0 or not math.isfinite(tolerance):
            raise ValueError("feasibility tolerance must be finite and nonnegative")
        return self.max_violation <= tolerance


def _restore_compiled_lp(
    matrix: sparse.spmatrix,
    rhs: ArrayLike,
    objective: ArrayLike,
    objective_offset: float,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    rows: tuple[RowRecord, ...],
    columns: tuple[ColumnRecord, ...],
    expected_fingerprint: str,
) -> CompiledLP:
    """Rebuild and integrity-check a model received by a worker process."""

    restored = CompiledLP(
        matrix,
        rhs,
        objective,
        objective_offset=objective_offset,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        rows=rows,
        columns=columns,
    )
    if restored.fingerprint != expected_fingerprint:
        raise ValueError("compiled LP fingerprint changed during process serialization")
    return restored


class CompiledLP:
    """Immutable canonical maximization LP with stable semantic metadata."""

    __slots__ = (
        "_column_lookup",
        "_columns",
        "_fingerprint",
        "_lower_bounds",
        "_matrix",
        "_objective",
        "_objective_offset",
        "_rhs",
        "_row_lookup",
        "_rows",
        "_upper_bounds",
    )

    def __init__(
        self,
        matrix: sparse.spmatrix | ArrayLike,
        rhs: ArrayLike,
        objective: ArrayLike,
        *,
        objective_offset: float = 0.0,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
        rows: Sequence[RowRecord] | None = None,
        columns: Sequence[ColumnRecord] | None = None,
    ) -> None:
        canonical = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
        if canonical.ndim != 2:
            raise ValueError("LP matrix must be two-dimensional")
        canonical.sum_duplicates()
        canonical.sort_indices()
        canonical.eliminate_zeros()
        canonical.data[canonical.data == 0.0] = 0.0
        if not np.all(np.isfinite(canonical.data)):
            raise ValueError("LP matrix coefficients must be finite")

        n_rows, n_columns = canonical.shape
        if n_columns == 0:
            raise ValueError("a compiled LP must contain at least one column")
        rhs_array = _readonly_float_array(rhs, name="rhs", length=n_rows)
        objective_array = _readonly_float_array(objective, name="objective", length=n_columns)
        offset = float(objective_offset)
        if not math.isfinite(offset):
            raise ValueError("objective offset must be finite")
        if offset == 0.0:
            offset = 0.0

        lower_array = _readonly_float_array(
            np.full(n_columns, -np.inf) if lower_bounds is None else lower_bounds,
            name="lower_bounds",
            length=n_columns,
            finite=False,
        )
        upper_array = _readonly_float_array(
            np.full(n_columns, np.inf) if upper_bounds is None else upper_bounds,
            name="upper_bounds",
            length=n_columns,
            finite=False,
        )
        if np.any(lower_array > upper_array):
            position = int(np.flatnonzero(lower_array > upper_array)[0])
            raise ValueError(f"column {position} has lower bound greater than upper bound")

        row_records = (
            tuple(rows)
            if rows is not None
            else tuple(
                RowRecord(key=f"row[{index}]", family="unlabeled") for index in range(n_rows)
            )
        )
        column_records = (
            tuple(columns)
            if columns is not None
            else tuple(
                ColumnRecord(key=f"z[{index}]", block="unlabeled") for index in range(n_columns)
            )
        )
        if len(row_records) != n_rows:
            raise ValueError("row metadata count does not match LP matrix")
        if len(column_records) != n_columns:
            raise ValueError("column metadata count does not match LP matrix")
        if any(not isinstance(row, RowRecord) for row in row_records):
            raise TypeError("rows must contain RowRecord objects")
        if any(not isinstance(column, ColumnRecord) for column in column_records):
            raise TypeError("columns must contain ColumnRecord objects")

        row_lookup = {record.key: index for index, record in enumerate(row_records)}
        column_lookup = {record.key: index for index, record in enumerate(column_records)}
        if len(row_lookup) != len(row_records):
            raise ValueError("compiled LP row keys must be unique")
        if len(column_lookup) != len(column_records):
            raise ValueError("compiled LP column keys must be unique")

        canonical.data.flags.writeable = False
        canonical.indices.flags.writeable = False
        canonical.indptr.flags.writeable = False

        self._matrix = canonical
        self._rhs = rhs_array
        self._objective = objective_array
        self._objective_offset = offset
        self._lower_bounds = lower_array
        self._upper_bounds = upper_array
        self._rows = row_records
        self._columns = column_records
        self._row_lookup = MappingProxyType(row_lookup)
        self._column_lookup = MappingProxyType(column_lookup)
        self._fingerprint = self._compute_fingerprint()

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Serialize through the validated constructor for process workers.

        The cached lookup tables are mapping proxies and are not pickled.
        Rebuilding them through :class:`CompiledLP` preserves the
        immutability checks and verifies that the numerical/semantic fingerprint
        survives the process boundary.
        """

        return (
            _restore_compiled_lp,
            (
                self._matrix,
                self._rhs,
                self._objective,
                self._objective_offset,
                self._lower_bounds,
                self._upper_bounds,
                self._rows,
                self._columns,
                self._fingerprint,
            ),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._matrix.shape[0]), int(self._matrix.shape[1]))

    @property
    def n_rows(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def n_columns(self) -> int:
        return int(self._matrix.shape[1])

    @property
    def nnz(self) -> int:
        return int(self._matrix.nnz)

    @property
    def matrix(self) -> sparse.csr_matrix:
        """Return a detached copy of the original-unit CSR matrix."""

        result = self._matrix.copy()
        result.data.flags.writeable = False
        result.indices.flags.writeable = False
        result.indptr.flags.writeable = False
        return result

    @property
    def rhs(self) -> FloatArray:
        return _view(self._rhs)

    @property
    def objective(self) -> FloatArray:
        return _view(self._objective)

    @property
    def objective_offset(self) -> float:
        return self._objective_offset

    @property
    def lower_bounds(self) -> FloatArray:
        return _view(self._lower_bounds)

    @property
    def upper_bounds(self) -> FloatArray:
        return _view(self._upper_bounds)

    @property
    def rows(self) -> tuple[RowRecord, ...]:
        return self._rows

    @property
    def columns(self) -> tuple[ColumnRecord, ...]:
        return self._columns

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 digest of numerical data and stable semantic IDs."""

        return self._fingerprint

    def row_index(self, key: str) -> int:
        return self._row_lookup[key]

    def column_index(self, key: str) -> int:
        return self._column_lookup[key]

    def objective_value(self, primal: ArrayLike) -> float:
        vector = _readonly_float_array(primal, name="primal", length=self.n_columns, finite=True)
        return float(np.dot(self._objective, vector) + self._objective_offset)

    def row_activity(self, primal: ArrayLike) -> FloatArray:
        vector = _readonly_float_array(primal, name="primal", length=self.n_columns, finite=True)
        activity = np.asarray(self._matrix @ vector, dtype=np.float64)
        activity.flags.writeable = False
        return activity

    def evaluate_primal(
        self, primal: ArrayLike, *, active_rows: ArrayLike | None = None
    ) -> PrimalResiduals:
        """Recompute row and bound violations without using solver residuals."""

        vector = _readonly_float_array(primal, name="primal", length=self.n_columns, finite=True)
        mask = _readonly_bool_array(
            np.ones(self.n_rows, dtype=np.bool_) if active_rows is None else active_rows,
            name="active_rows",
            length=self.n_rows,
        )
        activity = np.asarray(self._matrix @ vector, dtype=np.float64)
        slack = activity - self._rhs
        row_violation = np.empty(self.n_rows, dtype=np.float64)
        for index, record in enumerate(self._rows):
            if not mask[index]:
                row_violation[index] = 0.0
            elif record.sense is RowSense.GREATER_EQUAL:
                row_violation[index] = max(-float(slack[index]), 0.0)
            else:
                row_violation[index] = abs(float(slack[index]))

        lower_violation = np.maximum(self._lower_bounds - vector, 0.0)
        upper_violation = np.maximum(vector - self._upper_bounds, 0.0)
        max_row = max(0.0, float(np.max(row_violation))) if row_violation.size else 0.0
        max_bound = max(
            float(np.max(lower_violation)) if lower_violation.size else 0.0,
            float(np.max(upper_violation)) if upper_violation.size else 0.0,
        )
        for array in (activity, slack, row_violation, lower_violation, upper_violation):
            array[array == 0.0] = 0.0
            array.flags.writeable = False

        return PrimalResiduals(
            row_activity=activity,
            row_slack=slack,
            row_violation=row_violation,
            lower_bound_violation=lower_violation,
            upper_bound_violation=upper_violation,
            active_rows=mask,
            max_row_violation=max_row,
            max_bound_violation=max_bound,
            max_violation=max(max_row, max_bound),
        )

    def solver_data(self) -> SolverLPData:
        """Return a positively row-scaled CSC backend view."""

        factors = np.array([row.scale.factor for row in self._rows], dtype=np.float64)
        scaled = self._matrix.multiply(factors[:, np.newaxis]).tocsc()
        scaled.sum_duplicates()
        scaled.sort_indices()
        scaled.eliminate_zeros()
        scaled_rhs = factors * self._rhs
        if not np.all(np.isfinite(scaled.data)) or not np.all(np.isfinite(scaled_rhs)):
            raise ValueError("row scaling overflows the solver matrix or right-hand side")
        row_lower = np.empty(self.n_rows, dtype=np.float64)
        row_upper = np.empty(self.n_rows, dtype=np.float64)
        for index, record in enumerate(self._rows):
            row_lower[index] = scaled_rhs[index]
            row_upper[index] = scaled_rhs[index] if record.sense is RowSense.EQUAL else np.inf
        for array in (
            scaled.data,
            scaled.indices,
            scaled.indptr,
            row_lower,
            row_upper,
        ):
            array.flags.writeable = False
        return SolverLPData(
            matrix=scaled,
            objective=_view(self._objective),
            objective_offset=self._objective_offset,
            lower_bounds=_view(self._lower_bounds),
            upper_bounds=_view(self._upper_bounds),
            row_lower=_readonly_float_array(
                row_lower, name="solver row lower", length=self.n_rows, finite=False
            ),
            row_upper=_readonly_float_array(
                row_upper, name="solver row upper", length=self.n_rows, finite=False
            ),
        )

    def with_objective(self, objective: ArrayLike, *, offset: float | None = None) -> CompiledLP:
        """Return a copy with a new maximization objective."""

        return CompiledLP(
            self._matrix,
            self._rhs,
            objective,
            objective_offset=self._objective_offset if offset is None else offset,
            lower_bounds=self._lower_bounds,
            upper_bounds=self._upper_bounds,
            rows=self._rows,
            columns=self._columns,
        )

    def with_extra_rows(self, rows: Sequence[SparseRow]) -> CompiledLP:
        """Return a copy with sparse rows appended in declaration order."""

        additions = tuple(rows)
        if not additions:
            return self
        if any(not isinstance(row, SparseRow) for row in additions):
            raise TypeError("extra rows must be SparseRow objects")

        data: list[float] = []
        indices: list[int] = []
        indptr = [0]
        rhs = []
        records = []
        existing = set(self._row_lookup)
        for addition in additions:
            if addition.record.key in existing:
                raise ValueError(f"duplicate row key: {addition.record.key}")
            existing.add(addition.record.key)
            for column, coefficient in addition.expression:
                if column >= self.n_columns:
                    raise ValueError(
                        f"extra row {addition.record.key!r} references column {column} outside a {self.n_columns}-column model"
                    )
                indices.append(column)
                data.append(coefficient)
            indptr.append(len(data))
            rhs.append(addition.canonical_rhs)
            records.append(addition.record)

        extra_matrix = sparse.csr_matrix(
            (
                np.asarray(data, dtype=np.float64),
                np.asarray(indices, dtype=np.int64),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(len(additions), self.n_columns),
        )
        return CompiledLP(
            sparse.vstack((self._matrix, extra_matrix), format="csr"),
            np.concatenate((self._rhs, np.asarray(rhs, dtype=np.float64))),
            self._objective,
            objective_offset=self._objective_offset,
            lower_bounds=self._lower_bounds,
            upper_bounds=self._upper_bounds,
            rows=self._rows + tuple(records),
            columns=self._columns,
        )

    def with_rows_removed(self, rows: Iterable[int | str]) -> CompiledLP:
        """Return a structural copy with selected rows removed.

        Reusable backend sessions normally disable rows by replacing their
        bounds with infinities.  This method is useful for serialization and
        independent implication checks that require a physically smaller LP.
        """

        removed = set()
        for row in rows:
            index = self.row_index(row) if isinstance(row, str) else row
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("removed rows must be integer indices or stable keys")
            if index < 0 or index >= self.n_rows:
                raise IndexError(f"row index out of range: {index}")
            removed.add(index)
        if not removed:
            return self
        keep = np.array(
            [index for index in range(self.n_rows) if index not in removed], dtype=np.int64
        )
        return CompiledLP(
            self._matrix[keep, :],
            self._rhs[keep],
            self._objective,
            objective_offset=self._objective_offset,
            lower_bounds=self._lower_bounds,
            upper_bounds=self._upper_bounds,
            rows=tuple(self._rows[index] for index in keep),
            columns=self._columns,
        )

    def copy(self) -> CompiledLP:
        """Return an independent immutable copy."""

        return CompiledLP(
            self._matrix,
            self._rhs,
            self._objective,
            objective_offset=self._objective_offset,
            lower_bounds=self._lower_bounds,
            upper_bounds=self._upper_bounds,
            rows=self._rows,
            columns=self._columns,
        )

    def __copy__(self) -> CompiledLP:
        return self

    def __deepcopy__(self, memo: Mapping[int, Any]) -> CompiledLP:
        del memo
        return self

    def _compute_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"mechaudit.compiled-lp.v1\0")
        _hash_int(digest, self.n_rows)
        _hash_int(digest, self.n_columns)
        _hash_float_array(digest, self._matrix.data)
        _hash_int_array(digest, self._matrix.indices)
        _hash_int_array(digest, self._matrix.indptr)
        _hash_float_array(digest, self._rhs)
        _hash_float_array(digest, self._objective)
        _hash_float_array(digest, np.asarray([self._objective_offset]))
        _hash_float_array(digest, self._lower_bounds)
        _hash_float_array(digest, self._upper_bounds)
        for row in self._rows:
            _hash_string(digest, row.key)
            _hash_string(digest, row.family)
            _hash_string(digest, row.sense.value)
            _hash_float_array(digest, np.asarray([row.scale.factor, row.scale.report_factor]))
            digest.update(b"1" if row.classifiable else b"0")
        for column in self._columns:
            _hash_string(digest, column.key)
            _hash_string(digest, column.block)
        return digest.hexdigest()

    def __repr__(self) -> str:
        return f"CompiledLP(shape={self.shape!r}, nnz={self.nnz}, fingerprint={self.fingerprint[:12]!r})"


def _hash_int(digest: _HashLike, value: int) -> None:
    digest.update(struct.pack("<q", value))


def _hash_string(digest: _HashLike, value: str) -> None:
    encoded = value.encode("utf-8")
    _hash_int(digest, len(encoded))
    digest.update(encoded)


def _hash_float_array(digest: _HashLike, values: ArrayLike) -> None:
    array = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    array[array == 0.0] = 0.0
    little = array.astype("<f8", copy=False)
    _hash_int(digest, little.size)
    digest.update(little.tobytes(order="C"))


def _hash_int_array(digest: _HashLike, values: ArrayLike) -> None:
    little = np.asarray(values, dtype="<i8").reshape(-1)
    _hash_int(digest, little.size)
    digest.update(little.tobytes(order="C"))
