"""Transfer bounds and budget-balance requirements.

All classes use the package-wide sign convention: a positive payment is made
by an agent to the designer. Consequently, ``NoSubsidies`` means ``p >= 0``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any

from mechaudit.core.labels import ReportProfile
from mechaudit.exceptions import RequirementValidationError
from mechaudit.requirements.base import Requirement, Selector, SelectorLike, freeze_selector


class BudgetTiming(str, Enum):
    """Whether a budget requirement holds profile-wise or in expectation."""

    EX_POST = "ex_post"
    EX_ANTE = "ex_ante"


def _timing(value: Any) -> BudgetTiming:
    try:
        return BudgetTiming(value)
    except ValueError as exc:
        raise RequirementValidationError("budget timing must be 'ex_post' or 'ex_ante'") from exc


def _optional_bound(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, Real):
        raise RequirementValidationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise RequirementValidationError(f"{name} must be finite")
    return result


def _required_bound(value: Any, *, name: str) -> float:
    result = _optional_bound(value, name=name)
    if result is None:
        raise RequirementValidationError(f"{name} cannot be None")
    return result


def _reports(
    values: Iterable[ReportProfile] | None,
) -> tuple[ReportProfile, ...] | None:
    if values is None:
        return None
    result = tuple(values)
    if any(not isinstance(profile, ReportProfile) for profile in result):
        raise RequirementValidationError(
            "report selectors must contain canonical ReportProfile instances"
        )
    if len(set(result)) != len(result):
        raise RequirementValidationError("report selector contains duplicates")
    return result


@dataclass(frozen=True, init=False)
class PaymentBounds(Requirement):
    """Agent-by-report lower and/or upper payment bounds."""

    lower: float | None
    upper: float | None
    agents: Selector
    reports: tuple[ReportProfile, ...] | None

    def __init__(
        self,
        *,
        lower: float | None = None,
        upper: float | None = None,
        agents: SelectorLike = None,
        reports: Iterable[ReportProfile] | None = None,
    ) -> None:
        lower = _optional_bound(lower, name="lower")
        upper = _optional_bound(upper, name="upper")
        if lower is None and upper is None:
            raise RequirementValidationError("PaymentBounds requires at least one finite bound")
        if lower is not None and upper is not None and lower > upper:
            raise RequirementValidationError("payment lower bound exceeds upper bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "agents", freeze_selector(agents))
        object.__setattr__(self, "reports", _reports(reports))

    @property
    def family(self) -> str:
        return "payment_bounds"


@dataclass(frozen=True, init=False)
class NoSubsidies(Requirement):
    """Require every selected payment to be nonnegative (``p >= 0``)."""

    agents: Selector
    reports: tuple[ReportProfile, ...] | None

    def __init__(
        self,
        *,
        agents: SelectorLike = None,
        reports: Iterable[ReportProfile] | None = None,
    ) -> None:
        object.__setattr__(self, "agents", freeze_selector(agents))
        object.__setattr__(self, "reports", _reports(reports))

    @property
    def family(self) -> str:
        return "no_subsidies"


# Under the fixed payment-to-designer sign convention these phrases denote the
# same mathematical restriction. An alias prevents duplicate semantic families.
NoPositiveTransfers = NoSubsidies


@dataclass(frozen=True, init=False)
class WeakBudgetBalance(Requirement):
    """Require aggregate designer revenue to be at least ``floor``."""

    timing: BudgetTiming
    floor: float

    def __init__(
        self,
        *,
        timing: BudgetTiming | str = BudgetTiming.EX_POST,
        floor: float = 0.0,
    ) -> None:
        object.__setattr__(self, "timing", _timing(timing))
        object.__setattr__(self, "floor", _required_bound(floor, name="floor"))

    @property
    def family(self) -> str:
        return "weak_budget_balance"


@dataclass(frozen=True, init=False)
class StrongBudgetBalance(Requirement):
    """Require aggregate designer revenue to equal ``target``."""

    timing: BudgetTiming
    target: float

    def __init__(
        self,
        *,
        timing: BudgetTiming | str = BudgetTiming.EX_POST,
        target: float = 0.0,
    ) -> None:
        object.__setattr__(self, "timing", _timing(timing))
        object.__setattr__(self, "target", _required_bound(target, name="target"))

    @property
    def family(self) -> str:
        return "strong_budget_balance"


def PaymentFloor(
    floor: float,
    *,
    agents: Iterable[Any] | None = None,
    reports: Iterable[ReportProfile] | None = None,
) -> PaymentBounds:
    """Convenience constructor for a payment lower bound."""

    return PaymentBounds(lower=floor, agents=agents, reports=reports)


def PaymentCap(
    cap: float,
    *,
    agents: Iterable[Any] | None = None,
    reports: Iterable[ReportProfile] | None = None,
) -> PaymentBounds:
    """Convenience constructor for a payment upper bound."""

    return PaymentBounds(upper=cap, agents=agents, reports=reports)
