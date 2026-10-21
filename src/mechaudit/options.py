"""Numerical and workflow option objects.

Tolerances are separated by semantic role. A single generic
epsilon is not adequate for feasibility, activity, optimality, and optimal-
partition classification.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, cast

from mechaudit.core.labels import freeze_label
from mechaudit.exceptions import ValidationError


def _nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValidationError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class TolerancePolicy:
    """Numerical tolerances kept in their distinct economic roles."""

    probability: float = 1e-12
    feasibility: float = 1e-8
    activity: float = 1e-8
    objective_absolute: float = 1e-9
    objective_relative: float = 1e-9
    partition_primal: float = 1e-9
    partition_dual: float = 1e-9
    complementarity: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "probability",
            "feasibility",
            "activity",
            "objective_absolute",
            "objective_relative",
            "partition_primal",
            "partition_dual",
            "complementarity",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name=name))

    def objective_delta(self, objective_value: Real) -> float:
        """Return the effective absolute width of the near-optimal face."""

        checked_value = cast(Any, objective_value)
        if (
            isinstance(checked_value, bool)
            or not isinstance(checked_value, Real)
            or not math.isfinite(float(objective_value))
        ):
            raise ValidationError("objective_value must be finite")
        scale = max(1.0, abs(float(objective_value)))
        return max(self.objective_absolute, self.objective_relative * scale)


@dataclass(frozen=True)
class AuditOptions:
    """Options for fixed-mechanism economic auditing."""

    tolerances: TolerancePolicy = field(default_factory=TolerancePolicy)
    max_witnesses: int = 10
    include_all_rows: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.tolerances, TolerancePolicy):
            raise ValidationError("tolerances must be a TolerancePolicy")
        if (
            isinstance(self.max_witnesses, bool)
            or not isinstance(self.max_witnesses, int)
            or self.max_witnesses < 0
        ):
            raise ValidationError("max_witnesses must be a nonnegative integer")
        if not isinstance(self.include_all_rows, bool):
            raise ValidationError("include_all_rows must be a boolean")


class SolveMethod(str, Enum):
    """Backend-independent principal LP method preference."""

    AUTO = "auto"
    SIMPLEX = "simplex"
    IPM = "ipm"


@dataclass(frozen=True)
class SolveOptions:
    """Backend-independent solve options plus an explicit backend escape hatch."""

    tolerances: TolerancePolicy = field(default_factory=TolerancePolicy)
    method: SolveMethod = SolveMethod.AUTO
    crossover: bool | None = None
    presolve: bool = True
    time_limit: float | None = None
    threads: int | None = None
    random_seed: int | None = None
    backend_options: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tolerances, TolerancePolicy):
            raise ValidationError("tolerances must be a TolerancePolicy")
        try:
            object.__setattr__(self, "method", SolveMethod(self.method))
        except ValueError as exc:
            raise ValidationError("method must be auto, simplex, or ipm") from exc
        if self.crossover is not None and not isinstance(self.crossover, bool):
            raise ValidationError("crossover must be a boolean or None")
        if not isinstance(self.presolve, bool):
            raise ValidationError("presolve must be a boolean")
        if self.time_limit is not None:
            limit = _nonnegative(self.time_limit, name="time_limit")
            if limit == 0.0:
                raise ValidationError("time_limit must be positive when provided")
            object.__setattr__(self, "time_limit", limit)
        if self.threads is not None and (
            isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads <= 0
        ):
            raise ValidationError("threads must be a positive integer")
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int)
        ):
            raise ValidationError("random_seed must be an integer")
        raw_options = (
            self.backend_options.items()
            if isinstance(self.backend_options, Mapping)
            else self.backend_options
        )
        options = tuple((key, freeze_label(value)) for key, value in raw_options)
        if any(not isinstance(key, str) or not key for key, _ in options):
            raise ValidationError("backend option names must be nonempty strings")
        if len({key for key, _ in options}) != len(options):
            raise ValidationError("backend_options contains duplicate names")
        object.__setattr__(self, "backend_options", options)


class OptimalFaceLevel(str, Enum):
    """Cost tiers for optimal-face analysis."""

    FAST = "fast"
    FULL = "full"
    ESSENTIALITY = "essentiality"


@dataclass(frozen=True)
class OptimalFaceOptions:
    """Options for universal/nonuniversal and full activity analysis."""

    level: OptimalFaceLevel = OptimalFaceLevel.FAST
    tolerances: TolerancePolicy = field(default_factory=TolerancePolicy)
    objective_delta: float | None = None
    dual_ranges: bool = False
    constraint_families: tuple[str, ...] | None = None
    n_jobs: int = 1

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "level", OptimalFaceLevel(self.level))
        except ValueError as exc:
            raise ValidationError("level must be fast, full, or essentiality") from exc
        if not isinstance(self.tolerances, TolerancePolicy):
            raise ValidationError("tolerances must be a TolerancePolicy")
        if not isinstance(self.dual_ranges, bool):
            raise ValidationError("dual_ranges must be a boolean")
        if self.objective_delta is not None:
            object.__setattr__(
                self,
                "objective_delta",
                _nonnegative(self.objective_delta, name="objective_delta"),
            )
        if self.constraint_families is not None:
            raw_families = cast(Any, self.constraint_families)
            if isinstance(raw_families, (str, bytes)):
                raise ValidationError("constraint_families must contain nonempty strings")
            families = tuple(raw_families)
            if any(not isinstance(family, str) or not family for family in families):
                raise ValidationError("constraint_families must contain nonempty strings")
            object.__setattr__(self, "constraint_families", families)
        if (
            isinstance(self.n_jobs, bool)
            or not isinstance(self.n_jobs, int)
            or self.n_jobs == 0
            or self.n_jobs < -1
        ):
            raise ValidationError("n_jobs must be -1 or a positive integer")

    def effective_delta(self, objective_value: Real) -> float:
        """Return the explicitly requested or policy-derived face width."""

        if self.objective_delta is not None:
            return self.objective_delta
        return self.tolerances.objective_delta(objective_value)
