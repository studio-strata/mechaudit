"""Optimization backend interfaces and the direct HiGHS adapter."""

from mechaudit.backends.base import (
    BackendError,
    BackendFeatureError,
    BackendOperationError,
    BackendSession,
    BackendUnavailableError,
    BasisSnapshot,
    LPBackend,
    ModelStatus,
    SolverConfig,
    SolveResult,
)
from mechaudit.backends.highs import HighsBackend, HighsSession

__all__ = [
    "BackendError",
    "BackendFeatureError",
    "BackendOperationError",
    "BackendSession",
    "BackendUnavailableError",
    "BasisSnapshot",
    "HighsBackend",
    "HighsSession",
    "LPBackend",
    "ModelStatus",
    "SolveResult",
    "SolverConfig",
]
