"""Import helpers for labeled direct mechanisms.

This module performs no economic feasibility checks.  Its job is
to turn user-facing mappings, functions, arrays, pandas tables, and xarray
objects into finite, correctly aligned ``float64`` matrices.  Negative or
non-normalized allocation rows remain data for the audit layer to diagnose.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from mechaudit.core.labels import AgentId, OutcomeValue, ReportProfile, freeze_label
from mechaudit.exceptions import DomainMismatchError, ValidationError

FloatArray = NDArray[np.float64]


class EnvironmentDomain(Protocol):
    """The domain-only environment interface used by mechanism importers."""

    @property
    def agents(self) -> tuple[AgentId, ...]: ...

    @property
    def outcomes(self) -> tuple[OutcomeValue, ...]: ...

    @property
    def report_profiles(self) -> tuple[ReportProfile, ...]: ...

    @property
    def domain_fingerprint(self) -> str: ...

    @property
    def n_agents(self) -> int: ...

    @property
    def n_profiles(self) -> int: ...

    @property
    def n_outcomes(self) -> int: ...

    def coerce_report_profile(self, report: Any) -> ReportProfile: ...

    def profile_index(self, profile: Any) -> int: ...

    def outcome_index(self, outcome: Any) -> int: ...


@dataclass(frozen=True)
class MappingProvider:
    """A shallow immutable snapshot of a profile-indexed mapping."""

    entries: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True)
class FunctionProvider:
    """A callback evaluated only once the mechanism is bound to a domain."""

    function: Callable[..., Any]


@dataclass(frozen=True)
class ArrayProvider:
    """A numeric array with optional explicit row and column coordinates."""

    values: FloatArray
    row_labels: tuple[Any, ...] | None = None
    column_labels: tuple[Any, ...] | None = None


Provider = MappingProvider | FunctionProvider | ArrayProvider


@dataclass(frozen=True)
class MechanismSource:
    """Unbound allocation/payment providers and optional domain provenance."""

    allocation: Provider
    payments: Provider
    domain_fingerprint: str | None = None


def _finite_float(value: Any, *, path: str) -> float:
    if isinstance(value, (str, bytes, bytearray, complex)):
        raise ValidationError("mechanism entries must be real numbers", path=path)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("mechanism entries must be real numbers", path=path) from exc
    if not np.isfinite(result):
        raise ValidationError("mechanism entries must be finite", path=path)
    return result


def _numeric_array(value: Any, *, path: str, allow_one_dimensional: bool) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("mechanism array must contain real numbers", path=path) from exc
    valid_dimensions = (1, 2) if allow_one_dimensional else (2,)
    if array.ndim not in valid_dimensions:
        expected = "one or two dimensions" if allow_one_dimensional else "two dimensions"
        raise ValidationError(
            f"mechanism array must have {expected}",
            path=path,
            context={"shape": array.shape},
        )
    if not np.all(np.isfinite(array)):
        locations = np.argwhere(~np.isfinite(array))
        raise ValidationError(
            "mechanism array contains NaN or infinity",
            path=path,
            context={"first_invalid_index": tuple(int(i) for i in locations[0])},
        )
    return np.array(array, dtype=np.float64, order="C", copy=True)


def mapping_provider(value: Mapping[Any, Any], *, path: str) -> MappingProvider:
    """Snapshot a nested profile mapping and validate known numeric leaves."""

    if not isinstance(value, Mapping):
        raise ValidationError("expected a profile-indexed mapping", path=path)
    entries = []
    for row_index, (profile, raw_row) in enumerate(value.items()):
        row_path = f"{path}[{row_index}]"
        row: Any
        if isinstance(raw_row, Mapping):
            row = tuple(
                (coordinate, _finite_float(cell, path=f"{row_path}[{coordinate!r}]"))
                for coordinate, cell in raw_row.items()
            )
        else:
            # A scalar row is accepted later only for one-agent payments.  It
            # is still validated now so an unbound constructor rejects NaN/Inf.
            row = _finite_float(raw_row, path=row_path)
        entries.append((profile, row))
    return MappingProvider(tuple(entries))


def array_provider(
    value: Any,
    *,
    path: str,
    row_labels: Sequence[Any] | None = None,
    column_labels: Sequence[Any] | None = None,
    allow_one_dimensional: bool = False,
) -> ArrayProvider:
    """Create a validated labeled-array provider."""

    array = _numeric_array(value, path=path, allow_one_dimensional=allow_one_dimensional)
    rows = tuple(row_labels) if row_labels is not None else None
    columns = tuple(column_labels) if column_labels is not None else None
    if rows is not None and len(rows) != array.shape[0]:
        raise ValidationError(
            "row coordinate count does not match the array",
            path=path,
            context={"coordinates": len(rows), "rows": array.shape[0]},
        )
    column_count = 1 if array.ndim == 1 else array.shape[1]
    if columns is not None and len(columns) != column_count:
        raise ValidationError(
            "column coordinate count does not match the array",
            path=path,
            context={"coordinates": len(columns), "columns": column_count},
        )
    return ArrayProvider(array, rows, columns)


def provider_from_raw(value: Any, *, kind: str) -> Provider:
    """Normalize a raw constructor value to an immutable provider."""

    if isinstance(value, (MappingProvider, FunctionProvider, ArrayProvider)):
        return value
    if isinstance(value, Mapping):
        return mapping_provider(value, path=kind)
    if callable(value):
        return FunctionProvider(value)
    return array_provider(
        value,
        path=kind,
        allow_one_dimensional=kind == "payments",
    )


def mechanism_source(
    allocation: Any,
    payments: Any,
    *,
    domain_fingerprint: str | None = None,
) -> MechanismSource:
    """Create an unbound source from mixed raw provider styles."""

    if domain_fingerprint is not None and not isinstance(domain_fingerprint, str):
        raise ValidationError("domain_fingerprint must be a string or None")
    return MechanismSource(
        provider_from_raw(allocation, kind="allocation"),
        provider_from_raw(payments, kind="payments"),
        domain_fingerprint,
    )


def _agent_index(environment: EnvironmentDomain, raw_agent: Any) -> int:
    candidate = freeze_label(raw_agent, path="payments.agent")
    for index, agent in enumerate(environment.agents):
        if candidate == agent:
            return index
    raise DomainMismatchError("unknown payment agent", context={"agent": candidate})


def _coerce_row_profile(
    environment: EnvironmentDomain,
    raw_profile: Any,
    *,
    path: str,
) -> tuple[ReportProfile, int]:
    try:
        profile = environment.coerce_report_profile(raw_profile)
        index = environment.profile_index(profile)
    except (ValidationError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, DomainMismatchError):
            raise
        raise DomainMismatchError(
            "invalid or out-of-domain report profile",
            path=path,
            context={"profile": raw_profile},
        ) from exc
    return profile, index


def _mapping_rows(
    provider: MappingProvider,
    environment: EnvironmentDomain,
    *,
    kind: str,
) -> FloatArray:
    column_count = environment.n_outcomes if kind == "allocation" else environment.n_agents
    result = np.empty((environment.n_profiles, column_count), dtype=np.float64)
    seen_rows: dict[int, Any] = {}

    for entry_index, (raw_profile, raw_row) in enumerate(provider.entries):
        row_path = f"{kind}[{entry_index}]"
        profile, row_index = _coerce_row_profile(environment, raw_profile, path=row_path)
        if row_index in seen_rows:
            raise ValidationError(
                "duplicate report profile after domain coercion",
                path=row_path,
                context={"profile": profile, "first_key": seen_rows[row_index]},
            )
        seen_rows[row_index] = raw_profile

        if kind == "payments" and not isinstance(raw_row, tuple):
            if environment.n_agents != 1:
                raise ValidationError(
                    "scalar payment rows are valid only in one-agent environments",
                    path=row_path,
                )
            result[row_index, 0] = _finite_float(raw_row, path=row_path)
            continue
        if not isinstance(raw_row, tuple):
            raise ValidationError(
                "allocation rows must map every outcome to a probability",
                path=row_path,
            )

        seen_columns: dict[int, Any] = {}
        for raw_coordinate, value in raw_row:
            try:
                column_index = (
                    environment.outcome_index(raw_coordinate)
                    if kind == "allocation"
                    else _agent_index(environment, raw_coordinate)
                )
            except (ValidationError, KeyError, TypeError, ValueError) as exc:
                label = "outcome" if kind == "allocation" else "agent"
                raise DomainMismatchError(
                    f"unknown {label} coordinate",
                    path=f"{row_path}[{raw_coordinate!r}]",
                    context={label: raw_coordinate},
                ) from exc
            if column_index in seen_columns:
                raise ValidationError(
                    "duplicate column after label normalization",
                    path=row_path,
                    context={
                        "coordinate": raw_coordinate,
                        "first_coordinate": seen_columns[column_index],
                    },
                )
            seen_columns[column_index] = raw_coordinate
            result[row_index, column_index] = _finite_float(
                value, path=f"{row_path}[{raw_coordinate!r}]"
            )

        missing_columns = [
            (environment.outcomes if kind == "allocation" else environment.agents)[index]
            for index in range(column_count)
            if index not in seen_columns
        ]
        if missing_columns:
            raise ValidationError(
                "{} row does not cover the complete {} axis".format(
                    kind, "outcome" if kind == "allocation" else "agent"
                ),
                path=row_path,
                context={"missing": tuple(missing_columns)},
            )

    missing_rows = [
        environment.report_profiles[index]
        for index in range(environment.n_profiles)
        if index not in seen_rows
    ]
    if missing_rows:
        raise ValidationError(
            f"{kind} mapping does not cover every report profile",
            path=kind,
            context={"missing_count": len(missing_rows), "examples": tuple(missing_rows[:3])},
        )
    return result


def _signature_accepts(function: Callable[..., Any], argument_count: int) -> bool | None:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return None
    placeholders = (object(),) * argument_count
    try:
        signature.bind(*placeholders)
    except TypeError:
        return False
    return True


def _function_mode(function: Callable[..., Any], *, kind: str) -> str:
    accepts_one = _signature_accepts(function, 1)
    accepts_two = _signature_accepts(function, 2)
    if accepts_one is True and accepts_two is not True:
        return "row"
    if accepts_two is True and accepts_one is not True:
        return "cell"
    if accepts_one is True and accepts_two is True:
        # Optional second arguments and ``*args`` use the more explicit cell
        # convention.  Users wanting a row callback can declare one argument.
        return "cell"
    if accepts_one is False and accepts_two is False:
        raise ValidationError(
            f"{kind} callback must accept (report) or (report, coordinate)",
            path=kind,
        )
    # Some extension callables do not expose a signature.  The row convention
    # is least surprising and produces a semantic error if its result is not a row.
    return "row"


def _row_from_callback(
    raw_row: Any,
    environment: EnvironmentDomain,
    *,
    kind: str,
    profile: ReportProfile,
) -> FloatArray:
    if kind == "payments" and environment.n_agents == 1 and not isinstance(raw_row, Mapping):
        return np.asarray(
            [_finite_float(raw_row, path=f"payments[{profile!r}]")],
            dtype=np.float64,
        )
    if not isinstance(raw_row, Mapping):
        raise ValidationError(
            f"{kind} row callback must return a complete labeled mapping",
            path=f"{kind}[{profile!r}]",
        )
    provider = mapping_provider({profile: raw_row}, path=kind)
    # Reuse the mapping row parser by supplying placeholder rows for all other
    # profiles would be wasteful.  Parse the sole row through a tiny local loop.
    column_count = environment.n_outcomes if kind == "allocation" else environment.n_agents
    row = np.empty(column_count, dtype=np.float64)
    seen: set[int] = set()
    only_row = provider.entries[0][1]
    assert isinstance(only_row, tuple)
    for raw_coordinate, value in only_row:
        try:
            index = (
                environment.outcome_index(raw_coordinate)
                if kind == "allocation"
                else _agent_index(environment, raw_coordinate)
            )
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            raise DomainMismatchError(
                "unknown {} callback coordinate".format(
                    "outcome" if kind == "allocation" else "agent"
                ),
                context={"coordinate": raw_coordinate, "profile": profile},
            ) from exc
        if index in seen:
            raise ValidationError(
                "callback row contains duplicate normalized coordinates",
                context={"profile": profile, "coordinate": raw_coordinate},
            )
        seen.add(index)
        row[index] = value
    if len(seen) != column_count:
        axis = environment.outcomes if kind == "allocation" else environment.agents
        raise ValidationError(
            "callback row does not cover its complete labeled axis",
            context={
                "profile": profile,
                "missing": tuple(axis[index] for index in range(column_count) if index not in seen),
            },
        )
    return row


def _function_rows(
    provider: FunctionProvider,
    environment: EnvironmentDomain,
    *,
    kind: str,
) -> FloatArray:
    function = provider.function
    mode = _function_mode(function, kind=kind)
    column_axis = environment.outcomes if kind == "allocation" else environment.agents
    result = np.empty((environment.n_profiles, len(column_axis)), dtype=np.float64)
    for row_index, profile in enumerate(environment.report_profiles):
        if mode == "row":
            try:
                raw_row = function(profile)
            except Exception as exc:
                raise ValidationError(
                    f"{kind} callback failed",
                    path=f"{kind}[{profile!r}]",
                ) from exc
            result[row_index, :] = _row_from_callback(
                raw_row, environment, kind=kind, profile=profile
            )
            continue
        for column_index, coordinate in enumerate(column_axis):
            try:
                value = function(profile, coordinate)
            except Exception as exc:
                raise ValidationError(
                    f"{kind} callback failed",
                    path=f"{kind}[{profile!r}, {coordinate!r}]",
                ) from exc
            result[row_index, column_index] = _finite_float(
                value,
                path=f"{kind}[{profile!r}, {coordinate!r}]",
            )
    return result


def _array_rows(
    provider: ArrayProvider,
    environment: EnvironmentDomain,
    *,
    kind: str,
) -> FloatArray:
    values = provider.values
    if kind == "payments" and values.ndim == 1:
        if environment.n_agents != 1:
            raise ValidationError(
                "one-dimensional payments require a one-agent environment",
                path="payments",
                context={"shape": values.shape},
            )
        values = values.reshape((-1, 1))
    if values.ndim != 2:
        raise ValidationError("mechanism arrays must be two-dimensional after binding")

    expected_columns = environment.n_outcomes if kind == "allocation" else environment.n_agents
    if values.shape != (environment.n_profiles, expected_columns):
        raise ValidationError(
            f"{kind} array shape does not match the environment",
            path=kind,
            context={
                "shape": values.shape,
                "expected": (environment.n_profiles, expected_columns),
            },
        )

    row_order = list(range(environment.n_profiles))
    if provider.row_labels is not None:
        row_order = [-1] * environment.n_profiles
        for source_index, raw_profile in enumerate(provider.row_labels):
            _, target_index = _coerce_row_profile(
                environment,
                raw_profile,
                path=f"{kind}.rows[{source_index}]",
            )
            if row_order[target_index] != -1:
                raise ValidationError(
                    "duplicate labeled report row after domain coercion",
                    path=f"{kind}.rows",
                    context={"profile": raw_profile},
                )
            row_order[target_index] = source_index
        if any(index < 0 for index in row_order):
            raise ValidationError(
                "labeled array does not cover every report profile",
                path=f"{kind}.rows",
            )

    column_order = list(range(expected_columns))
    if provider.column_labels is not None:
        column_order = [-1] * expected_columns
        for source_index, raw_coordinate in enumerate(provider.column_labels):
            try:
                target_index = (
                    environment.outcome_index(raw_coordinate)
                    if kind == "allocation"
                    else _agent_index(environment, raw_coordinate)
                )
            except (ValidationError, KeyError, TypeError, ValueError) as exc:
                raise DomainMismatchError(
                    f"labeled {kind} array contains an unknown column",
                    context={"coordinate": raw_coordinate},
                ) from exc
            if column_order[target_index] != -1:
                raise ValidationError(
                    "duplicate labeled column after domain coercion",
                    path=f"{kind}.columns",
                    context={"coordinate": raw_coordinate},
                )
            column_order[target_index] = source_index
        if any(index < 0 for index in column_order):
            raise ValidationError(
                "labeled array does not cover its complete column axis",
                path=f"{kind}.columns",
            )

    return np.array(values[np.ix_(row_order, column_order)], dtype=np.float64, order="C")


def canonicalize_source(
    source: MechanismSource,
    environment: EnvironmentDomain,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate and align an unbound source against *environment*."""

    if source.domain_fingerprint is not None and (
        source.domain_fingerprint != environment.domain_fingerprint
    ):
        raise DomainMismatchError(
            "mechanism domain fingerprint does not match the environment",
            context={
                "mechanism": source.domain_fingerprint,
                "environment": environment.domain_fingerprint,
            },
        )

    def convert(provider: Provider, kind: str) -> FloatArray:
        if isinstance(provider, MappingProvider):
            return _mapping_rows(provider, environment, kind=kind)
        if isinstance(provider, FunctionProvider):
            return _function_rows(provider, environment, kind=kind)
        return _array_rows(provider, environment, kind=kind)

    allocation = convert(source.allocation, "allocation")
    payments = convert(source.payments, "payments")
    # Callback results are checked cell-by-cell, but this final check also
    # protects future provider implementations.
    if not np.all(np.isfinite(allocation)) or not np.all(np.isfinite(payments)):
        raise ValidationError("bound mechanism contains NaN or infinity")
    return allocation, payments


def _table_axis(table: Any, attribute: str, *, path: str) -> tuple[Any, ...]:
    try:
        axis = getattr(table, attribute)
        if hasattr(axis, "is_unique") and not bool(axis.is_unique):
            raise ValidationError("table axis contains duplicate labels", path=path)
        return tuple(axis.tolist())
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("object is not a supported pandas-like table", path=path) from exc


def source_from_tables(allocation: Any, payments: Any) -> MechanismSource:
    """Build a source from two wide pandas-like tables.

    Rows are report profiles; allocation columns are outcomes and payment
    columns are agents.  A pandas ``MultiIndex`` naturally supplies tuple
    report shorthands for multi-agent environments.
    """

    try:
        allocation_values = allocation.to_numpy()
        payment_values = payments.to_numpy()
    except Exception as exc:
        raise ValidationError("allocation and payments must be pandas-like tables") from exc

    signs = {
        value
        for value in (
            getattr(allocation, "attrs", {}).get("payment_sign"),
            getattr(payments, "attrs", {}).get("payment_sign"),
        )
        if value is not None
    }
    if len(signs) > 1:
        raise ValidationError("allocation and payment tables declare different payment signs")
    if signs and next(iter(signs)) != "agent_to_designer":
        raise ValidationError(
            "unsupported table payment sign convention",
            context={"payment_sign": next(iter(signs))},
        )

    allocation_provider = array_provider(
        allocation_values,
        path="allocation",
        row_labels=_table_axis(allocation, "index", path="allocation.index"),
        column_labels=_table_axis(allocation, "columns", path="allocation.columns"),
    )
    payment_provider = array_provider(
        payment_values,
        path="payments",
        row_labels=_table_axis(payments, "index", path="payments.index"),
        column_labels=_table_axis(payments, "columns", path="payments.columns"),
        allow_one_dimensional=True,
    )
    fingerprints = {
        value
        for value in (
            getattr(allocation, "attrs", {}).get("domain_fingerprint"),
            getattr(payments, "attrs", {}).get("domain_fingerprint"),
        )
        if value is not None
    }
    if len(fingerprints) > 1:
        raise DomainMismatchError("allocation and payment tables have different fingerprints")
    fingerprint = next(iter(fingerprints), None)
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise ValidationError("table domain_fingerprint metadata must be a string")
    return MechanismSource(allocation_provider, payment_provider, fingerprint)


def _xarray_coordinate(data: Any, dimension: Any, *, path: str) -> tuple[Any, ...]:
    try:
        coordinate = data.coords[dimension]
        return tuple(coordinate.values.tolist())
    except Exception as exc:
        raise ValidationError(
            "xarray dimension must have an explicit coordinate",
            path=path,
            context={"dimension": dimension},
        ) from exc


def _xarray_provider(
    data: Any,
    *,
    kind: str,
    column_dim: Any,
    report_dims: Sequence[Any] | None,
) -> ArrayProvider:
    if not all(hasattr(data, name) for name in ("dims", "coords", "values", "transpose")):
        raise ValidationError("object is not an xarray-like DataArray", path=kind)
    dimensions = tuple(data.dims)
    if column_dim not in dimensions:
        raise ValidationError(
            "xarray data is missing its declared column dimension",
            path=kind,
            context={"dimension": column_dim, "dims": dimensions},
        )
    rows = (
        tuple(report_dims)
        if report_dims is not None
        else tuple(dimension for dimension in dimensions if dimension != column_dim)
    )
    if not rows:
        raise ValidationError(
            "xarray mechanism data needs at least one report dimension", path=kind
        )
    if len(set(rows)) != len(rows) or set(rows).union({column_dim}) != set(dimensions):
        raise ValidationError(
            "report and column dimensions must cover each xarray dimension exactly once",
            path=kind,
            context={"report_dims": rows, "column_dim": column_dim, "dims": dimensions},
        )
    try:
        ordered = data.transpose(*rows, column_dim)
    except Exception as exc:
        raise ValidationError("could not transpose xarray mechanism data", path=kind) from exc

    row_coordinates = [
        _xarray_coordinate(ordered, dimension, path=f"{kind}.coords") for dimension in rows
    ]
    row_labels = row_coordinates[0] if len(rows) == 1 else tuple(product(*row_coordinates))
    column_labels = _xarray_coordinate(ordered, column_dim, path=f"{kind}.coords")
    values = np.asarray(ordered.values)
    flattened = values.reshape((len(row_labels), len(column_labels)))
    return array_provider(
        flattened,
        path=kind,
        row_labels=row_labels,
        column_labels=column_labels,
    )


def source_from_xarray(
    allocation: Any,
    payments: Any,
    *,
    report_dims: Sequence[Any] | None = None,
    outcome_dim: Any = "outcome",
    payment_agent_dim: Any = "agent",
) -> MechanismSource:
    """Build a source from labeled xarray-like allocation/payment arrays.

    Every dimension except the declared column dimension is a report axis.
    Multiple report axes are flattened in their declared order into tuple
    profile shorthands.
    """

    signs = {
        value
        for value in (
            getattr(allocation, "attrs", {}).get("payment_sign"),
            getattr(payments, "attrs", {}).get("payment_sign"),
        )
        if value is not None
    }
    if len(signs) > 1:
        raise ValidationError("allocation and payment arrays declare different payment signs")
    if signs and next(iter(signs)) != "agent_to_designer":
        raise ValidationError(
            "unsupported xarray payment sign convention",
            context={"payment_sign": next(iter(signs))},
        )

    allocation_provider = _xarray_provider(
        allocation,
        kind="allocation",
        column_dim=outcome_dim,
        report_dims=report_dims,
    )
    payment_provider = _xarray_provider(
        payments,
        kind="payments",
        column_dim=payment_agent_dim,
        report_dims=report_dims,
    )
    fingerprints = {
        value
        for value in (
            getattr(allocation, "attrs", {}).get("domain_fingerprint"),
            getattr(payments, "attrs", {}).get("domain_fingerprint"),
        )
        if value is not None
    }
    if len(fingerprints) > 1:
        raise DomainMismatchError("allocation and payment arrays have different fingerprints")
    fingerprint = next(iter(fingerprints), None)
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise ValidationError("xarray domain_fingerprint metadata must be a string")
    return MechanismSource(allocation_provider, payment_provider, fingerprint)


__all__ = ["MechanismSource"]
