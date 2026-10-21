"""Numerical-health data shared by audits and optimization results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NumericalHealthReport:
    """Original-unit residuals, scales, and economically labeled warnings."""

    coefficient_min_positive: float | None = None
    coefficient_max: float | None = None
    row_scale_min: float | None = None
    row_scale_max: float | None = None
    smallest_positive_prior: float | None = None
    smallest_type_marginal: float | None = None
    largest_allocation_residual: float = 0.0
    largest_normalization_residual: float = 0.0
    largest_ic_violation: float = 0.0
    largest_ir_violation: float = 0.0
    primal_residual: float | None = None
    dual_residual: float | None = None
    complementarity_residual: float | None = None
    payment_to_value_ratio: float | None = None
    condition_estimate: float | None = None
    warnings: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        """Whether the report emitted no numerical-health warning."""

        return not self.warnings

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {
            "coefficient_magnitude_range": [
                self.coefficient_min_positive,
                self.coefficient_max,
            ],
            "row_scaling_range": [self.row_scale_min, self.row_scale_max],
            "smallest_positive_prior_probability": self.smallest_positive_prior,
            "smallest_type_marginal": self.smallest_type_marginal,
            "largest_allocation_residual": self.largest_allocation_residual,
            "largest_probability_normalization_residual": self.largest_normalization_residual,
            "largest_ic_violation": self.largest_ic_violation,
            "largest_ir_violation": self.largest_ir_violation,
            "primal_residual": self.primal_residual,
            "dual_residual": self.dual_residual,
            "complementarity_residual": self.complementarity_residual,
            "payment_to_value_scale_ratio": self.payment_to_value_ratio,
            "condition_estimate": self.condition_estimate,
            "warnings": list(self.warnings),
            "details": dict(self.details),
        }

    def __str__(self) -> str:
        status = "healthy" if self.healthy else f"warnings: {len(self.warnings)}"
        return f"NumericalHealthReport({status})"
