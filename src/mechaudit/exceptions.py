"""Domain-specific exception hierarchy for :mod:`mechaudit`.

The library raises semantic exceptions with useful context instead of exposing
indexing errors from its internal array representation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class MechAuditError(Exception):
    """Base class for all public MechAudit exceptions."""


class ValidationError(MechAuditError, ValueError):
    """Base class for invalid user-supplied domain objects."""

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.path = path
        self.context = dict(context or {})
        prefix = f"{path}: " if path else ""
        super().__init__(prefix + message)


class LabelValidationError(ValidationError):
    """A label cannot be converted to a stable finite-domain label."""


class PriorValidationError(ValidationError):
    """A prior has invalid support, labels, or probabilities."""


class EnvironmentValidationError(ValidationError):
    """A finite environment is incomplete or internally inconsistent."""


class RequirementValidationError(ValidationError):
    """An economic requirement is malformed or incompatible with a model."""


class DomainMismatchError(ValidationError):
    """Objects use different agent, type, report, or outcome domains."""


class ZeroProbabilityTypeError(RequirementValidationError):
    """A conditional requirement encounters a zero-marginal type."""


class UnsupportedRequirementError(MechAuditError, NotImplementedError):
    """A backend or workflow does not support a declared requirement."""


class SolverError(MechAuditError):
    """Base class for solver and solution-certification failures."""


class InfeasibleDesignError(SolverError):
    """The declared design problem is infeasible."""


class UnboundedDesignError(SolverError):
    """The declared design objective is unbounded."""


class NumericalCertificationError(SolverError):
    """A numerical result could not be certified at requested tolerances."""
