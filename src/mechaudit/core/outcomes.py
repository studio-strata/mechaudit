"""Structured outcomes and small explicit outcome-space factories."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, overload

from mechaudit.core.labels import AgentId, Label, StructuredLabel, freeze_label, normalize_axis
from mechaudit.exceptions import LabelValidationError


@dataclass(frozen=True, init=False)
class Outcome(Mapping[str, Label]):
    """A readable, immutable structured social outcome.

    Examples
    --------
    ``Outcome(allocation={"item": "buyer"}, public_decision="build")``
    """

    _attributes: StructuredLabel

    def __init__(self, attributes: Mapping[str, Any] | None = None, **fields: Any) -> None:
        combined = dict(attributes or {})
        overlap = set(combined).intersection(fields)
        if overlap:
            raise LabelValidationError(
                "outcome attributes were supplied more than once",
                context={"attributes": sorted(overlap)},
            )
        combined.update(fields)
        if not combined:
            raise LabelValidationError("a structured outcome cannot be empty")
        object.__setattr__(self, "_attributes", StructuredLabel(combined))

    @classmethod
    def from_mapping(cls, attributes: Mapping[str, Any]) -> Outcome:
        """Create a structured outcome from an existing mapping."""

        return cls(attributes)

    def __getitem__(self, key: str) -> Label:
        return self._attributes[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._attributes)

    def __len__(self) -> int:
        return len(self._attributes)

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"Outcome({body})"

    def to_dict(self) -> dict[str, Any]:
        """Return a recursively plain mapping representation."""

        return self._attributes.to_dict()


@dataclass(frozen=True)
class OutcomeSpace(Sequence[Label]):
    """An explicitly enumerated, immutable outcome space."""

    outcomes: tuple[Label, ...]

    def __init__(self, outcomes: Iterable[Any]) -> None:
        object.__setattr__(self, "outcomes", normalize_axis(outcomes, name="outcomes"))

    @overload
    def __getitem__(self, index: int) -> Label: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Label, ...]: ...

    def __getitem__(self, index: int | slice) -> Label | tuple[Label, ...]:
        return self.outcomes[index]

    def __len__(self) -> int:
        return len(self.outcomes)

    def __iter__(self) -> Iterator[Label]:
        return iter(self.outcomes)


def single_item_outcomes(
    agents: Iterable[Any], *, include_unallocated: bool = True
) -> OutcomeSpace:
    """Return explicit single-item outcomes indexed by the winning agent."""

    agent_axis = normalize_axis(agents, name="agents")
    outcomes = []
    if include_unallocated:
        outcomes.append(Outcome(kind="single_item", winner=None))
    outcomes.extend(Outcome(kind="single_item", winner=agent) for agent in agent_axis)
    return OutcomeSpace(outcomes)


def multiple_item_outcomes(
    agents: Iterable[Any],
    items: Iterable[Any],
    *,
    allow_unallocated: bool = True,
) -> OutcomeSpace:
    """Enumerate assignments of explicit items to agents.

    This convenience enumerates the full finite assignment space and may grow
    exponentially. It is suitable for small examples, not implicit combinatorial domains.
    """

    agent_axis = normalize_axis(agents, name="agents")
    item_axis = normalize_axis(items, name="items")
    recipients: tuple[Any, ...] = agent_axis
    if allow_unallocated:
        recipients = (None, *recipients)
    outcomes = []
    for assignment in product(recipients, repeat=len(item_axis)):
        allocation = {
            item: recipient for item, recipient in zip(item_axis, assignment, strict=True)
        }
        outcomes.append(Outcome(kind="multiple_item", allocation=allocation))
    return OutcomeSpace(outcomes)


def public_project_outcomes() -> OutcomeSpace:
    """Return the two outcomes for a binary public-project decision."""

    return OutcomeSpace(
        (Outcome(kind="public_project", build=False), Outcome(kind="public_project", build=True))
    )


def bilateral_trade_outcomes() -> OutcomeSpace:
    """Return explicit trade and no-trade outcomes."""

    return OutcomeSpace(
        (Outcome(kind="bilateral_trade", trade=False), Outcome(kind="bilateral_trade", trade=True))
    )


def procurement_outcomes(
    suppliers: Iterable[AgentId], *, include_no_procurement: bool = True
) -> OutcomeSpace:
    """Return outcomes selecting at most one supplier."""

    supplier_axis = normalize_axis(suppliers, name="suppliers")
    outcomes = []
    if include_no_procurement:
        outcomes.append(Outcome(kind="procurement", supplier=None))
    outcomes.extend(
        Outcome(kind="procurement", supplier=freeze_label(supplier)) for supplier in supplier_axis
    )
    return OutcomeSpace(outcomes)
