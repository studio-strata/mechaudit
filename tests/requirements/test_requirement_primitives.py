"""Typed requirement declaration and validation contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from mechaudit import (
    AllocationCoordinate,
    Capacity,
    CustomLinearConstraint,
    CustomLinearRequirement,
    PaymentBounds,
    PaymentCap,
    PaymentCoordinate,
    PaymentFloor,
    ReportProfile,
    StrongBudgetBalance,
    Symmetry,
    SymmetryOperation,
    WeakBudgetBalance,
)
from mechaudit.exceptions import RequirementValidationError
from mechaudit.requirements.base import (
    ConstraintId,
    ConstraintSpec,
    LinearExpression,
    freeze_selector,
    freeze_selector_map,
)


class _RepeatedKeys(Mapping[Any, Any]):
    def __init__(self, keys: tuple[Any, ...], value: Any) -> None:
        self.keys_ = keys
        self.value_ = value

    def __getitem__(self, key: Any) -> Any:
        return self.value_

    def __iter__(self) -> Iterator[Any]:
        return iter(self.keys_)

    def __len__(self) -> int:
        return len(self.keys_)


def _report() -> ReportProfile:
    return ReportProfile((("buyer", "low"),))


def test_typed_coordinates_and_requirement_expression_algebra() -> None:
    report = _report()
    allocation = AllocationCoordinate(report, "sale")
    payment = PaymentCoordinate(report, "buyer")
    expression = LinearExpression(
        ((allocation, 2.0), (payment, -1.0), (allocation, -0.5)), constant=3.0
    )

    assert expression.terms == ((allocation, 1.5), (payment, -1.0))
    assert dict(expression.coefficients) == {allocation: 1.5, payment: -1.0}
    assert expression.scaled(2.0).constant == 6.0
    assert expression.plus(LinearExpression({payment: 1.0})).terms == ((allocation, 1.5),)
    assert expression.plus(object()) is NotImplemented  # type: ignore[arg-type]

    with pytest.raises(RequirementValidationError, match="ReportProfile"):
        AllocationCoordinate("low", "sale")  # type: ignore[arg-type]
    with pytest.raises(RequirementValidationError, match="ReportProfile"):
        PaymentCoordinate("low", "buyer")  # type: ignore[arg-type]
    with pytest.raises(RequirementValidationError, match="typed mechanism coordinates"):
        LinearExpression({"x": 1.0})  # type: ignore[dict-item]
    with pytest.raises(RequirementValidationError, match="real number"):
        LinearExpression({payment: True})
    with pytest.raises(RequirementValidationError, match="finite"):
        LinearExpression({payment: float("inf")})
    with pytest.raises(RequirementValidationError, match="factor"):
        expression.scaled(float("nan"))


def test_constraint_identity_specs_and_selectors() -> None:
    identifier = ConstraintId("  CUSTOM ", {"z": 2, "a": "x"})
    expression = LinearExpression()
    spec = ConstraintSpec(identifier, expression, relation="==", original_scale=2.0)

    assert identifier.family == "custom"
    assert identifier.qualifiers == (("a", "x"), ("z", 2))
    assert str(identifier) == "custom[a='x', z=2]"
    assert spec.relation.value == "=="
    assert str(ConstraintId("plain")) == "plain"
    assert freeze_selector(None) is None
    assert freeze_selector("buyer") == ("buyer",)
    assert freeze_selector(()) == ()
    assert freeze_selector_map(None) is None
    assert freeze_selector_map({"buyer": ("low", "high")}) == (("buyer", ("low", "high")),)

    with pytest.raises(RequirementValidationError, match="family"):
        ConstraintId("")
    with pytest.raises(RequirementValidationError, match="qualifier keys"):
        ConstraintId("x", {"": 1})
    with pytest.raises(RequirementValidationError, match="ConstraintId"):
        ConstraintSpec("x", expression)  # type: ignore[arg-type]
    with pytest.raises(RequirementValidationError, match="LinearExpression"):
        ConstraintSpec(identifier, object())  # type: ignore[arg-type]
    with pytest.raises(RequirementValidationError, match="relation"):
        ConstraintSpec(identifier, expression, relation="<=")  # type: ignore[arg-type]
    with pytest.raises(RequirementValidationError, match="positive"):
        ConstraintSpec(identifier, expression, original_scale=0.0)
    with pytest.raises(RequirementValidationError, match="duplicate labels"):
        freeze_selector(("buyer", "buyer"))
    with pytest.raises(RequirementValidationError, match="duplicate agents"):
        freeze_selector_map(_RepeatedKeys(("buyer", "buyer"), ("low",)))


def test_payment_and_budget_declarations_validate_signatures() -> None:
    report = _report()
    bounds = PaymentBounds(lower=-1, upper=2, agents="buyer", reports=(report,))

    assert bounds.family == "payment_bounds"
    assert bounds.agents == ("buyer",)
    assert PaymentFloor(0).lower == 0.0
    assert PaymentCap(3).upper == 3.0
    assert WeakBudgetBalance(timing="ex_ante", floor=-1).timing.value == "ex_ante"
    assert StrongBudgetBalance(target=2).target == 2.0

    with pytest.raises(RequirementValidationError, match="at least one"):
        PaymentBounds()
    with pytest.raises(RequirementValidationError, match="exceeds"):
        PaymentBounds(lower=2, upper=1)
    with pytest.raises(RequirementValidationError, match="real number"):
        PaymentBounds(lower="zero")  # type: ignore[arg-type]
    with pytest.raises(RequirementValidationError, match="finite"):
        PaymentBounds(upper=float("inf"))
    with pytest.raises(RequirementValidationError, match="ReportProfile"):
        PaymentBounds(lower=0, reports=("low",))  # type: ignore[arg-type]
    with pytest.raises(RequirementValidationError, match="duplicates"):
        PaymentBounds(lower=0, reports=(report, report))
    with pytest.raises(RequirementValidationError, match="budget timing"):
        WeakBudgetBalance(timing="interim")
    with pytest.raises(RequirementValidationError, match="cannot be None"):
        StrongBudgetBalance(target=None)  # type: ignore[arg-type]


def test_custom_groups_capacity_and_symmetry_validation() -> None:
    report = _report()
    payment = PaymentCoordinate(report, "buyer")
    row = CustomLinearConstraint(
        "cap", {payment: 1.0}, sense="<=", rhs=2.0, family="Policy", metadata={"tag": 1}
    )
    equality = CustomLinearConstraint("balance", {payment: 1.0}, sense="==", rhs=0.0)
    group = CustomLinearRequirement("rules", (row, equality))

    assert row.family == "policy"
    assert row.spec.expression.terms == ((payment, -1.0),)
    assert row.spec.expression.constant == 2.0
    assert equality.spec.relation.value == "=="
    assert group.family == "custom_group"

    for kwargs, message in (
        ({"name": "", "coefficients": {}}, "name"),
        ({"name": "x", "coefficients": {}, "family": ""}, "family"),
        ({"name": "x", "coefficients": {}, "sense": "!="}, "sense"),
        ({"name": "x", "coefficients": {}, "metadata": {"name": "reserved"}}, "reserved"),
    ):
        with pytest.raises(RequirementValidationError, match=message):
            CustomLinearConstraint(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(RequirementValidationError, match="name"):
        CustomLinearRequirement("", (row,))
    with pytest.raises(RequirementValidationError, match="at least one"):
        CustomLinearRequirement("empty", ())
    with pytest.raises(RequirementValidationError, match="CustomLinearConstraint"):
        CustomLinearRequirement("bad", (object(),))  # type: ignore[arg-type]
    with pytest.raises(RequirementValidationError, match="duplicate row ids"):
        CustomLinearRequirement("duplicate", (row, row))

    capacity = Capacity(resource="space", usage={"off": 0, "on": 1}, upper=1, lower=0)
    assert capacity.family == "capacity"
    with pytest.raises(RequirementValidationError, match="cannot be empty"):
        Capacity(resource="x", usage={}, upper=1)
    with pytest.raises(RequirementValidationError, match="real number"):
        Capacity(resource="x", usage={"on": "one"}, upper=1)  # type: ignore[dict-item]
    with pytest.raises(RequirementValidationError, match="finite"):
        Capacity(resource="x", usage={"on": 1}, upper=float("inf"))
    with pytest.raises(RequirementValidationError, match="exceeds"):
        Capacity(resource="x", usage={"on": 1}, lower=2, upper=1)

    operation = SymmetryOperation(
        agents={"buyer": "seller"},
        types={"buyer": {"low": "high"}},
        outcomes={"sale": "no_sale"},
        name="swap",
    )
    assert operation.name == "swap"
    assert Symmetry(operation).family == "symmetry"
    with pytest.raises(RequirementValidationError, match="name"):
        SymmetryOperation(agents={}, types={}, outcomes={}, name="")
    with pytest.raises(RequirementValidationError, match="at least one"):
        Symmetry()
    with pytest.raises(RequirementValidationError, match="SymmetryOperation"):
        Symmetry(object())  # type: ignore[arg-type]
