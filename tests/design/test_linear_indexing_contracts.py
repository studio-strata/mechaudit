"""Behavioral contracts for solver-independent linear and indexing primitives."""

from __future__ import annotations

import math

import pytest

from mechaudit.design.indexing import VariableIndex, VariableIndexBuilder, VariableKey
from mechaudit.design.linear import LinearExpression, LinearExpressionBuilder


def test_linear_expression_canonicalization_lookup_and_evaluation() -> None:
    expression = LinearExpression(
        ((2, 1e-14), (0, 2.0), (1, 4.0), (0, -0.5), (1, -4.0)),
        constant=-0.0,
    )

    assert expression.terms == ((0, 1.5), (2, 1e-14))
    assert expression.constant == 0.0
    assert math.copysign(1.0, expression.constant) == 1.0
    assert expression.indices == (0, 2)
    assert expression.coefficients == pytest.approx((1.5, 1e-14))
    assert expression.coefficient(0) == pytest.approx(1.5)
    assert expression.coefficient(1) == 0.0
    assert expression.coefficient(3) == 0.0
    assert tuple(expression) == expression.terms
    assert expression.evaluate((2.0, 99.0, 3.0)) == pytest.approx(3.0 + 3e-14)

    mapped = LinearExpression.from_mapping({3: 2, 0: -1}, constant=4)
    assert mapped.terms == ((0, -1.0), (3, 2.0))
    assert mapped.constant == 4.0


@pytest.mark.parametrize("column", (True, 1.5, "1"))
def test_linear_expression_rejects_noninteger_columns(column: object) -> None:
    with pytest.raises(TypeError, match="indices must be integers"):
        LinearExpression(((column, 1.0),))  # type: ignore[arg-type]


def test_linear_expression_rejects_invalid_numbers_and_vectors() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        LinearExpression(((-1, 1.0),))
    with pytest.raises(ValueError, match="coefficients must be finite"):
        LinearExpression(((0, float("nan")),))
    with pytest.raises(ValueError, match="constants must be finite"):
        LinearExpression(constant=float("inf"))
    with pytest.raises(ValueError, match="one-dimensional"):
        LinearExpression(((0, 1.0),)).evaluate(((1.0,), (2.0,)))
    with pytest.raises(ValueError, match="outside the supplied vector"):
        LinearExpression(((2, 1.0),)).evaluate((1.0, 2.0))
    with pytest.raises(ValueError, match="values must be finite"):
        LinearExpression(((0, 1.0),)).evaluate((float("inf"),))
    with pytest.raises(ValueError, match="scale must be finite"):
        LinearExpression(((0, 1.0),)).scaled(float("nan"))


def test_linear_expression_arithmetic_is_affine_and_type_safe() -> None:
    left = LinearExpression(((0, 2.0),), 3.0)
    right = LinearExpression(((0, -1.0), (1, 4.0)), -2.0)

    assert (left + right) == LinearExpression(((0, 1.0), (1, 4.0)), 1.0)
    assert (left + 2) == (2 + left) == LinearExpression(((0, 2.0),), 5.0)
    assert (left - right) == LinearExpression(((0, 3.0), (1, -4.0)), 5.0)
    assert (left - 2) == LinearExpression(((0, 2.0),), 1.0)
    assert (10 - left) == LinearExpression(((0, -2.0),), 7.0)
    assert (-left) == LinearExpression(((0, -2.0),), -3.0)
    assert (left * 2) == (2 * left) == LinearExpression(((0, 4.0),), 6.0)
    assert left.without_constant() == LinearExpression(((0, 2.0),))

    with pytest.raises(TypeError):
        _ = left + object()
    with pytest.raises(TypeError):
        _ = left - object()
    with pytest.raises(TypeError):
        _ = object() - left
    with pytest.raises(TypeError):
        _ = left * object()
    with pytest.raises(TypeError):
        _ = left + True


def test_linear_expression_builder_accumulates_and_clears() -> None:
    builder = LinearExpressionBuilder(constant=1)
    assert builder.add_term(0, 2) is builder
    assert builder.add_term(0, -0.5) is builder
    assert builder.add_expression(LinearExpression(((1, 3.0),), 2.0), factor=2) is builder
    assert builder.add_constant(-1) is builder
    assert builder.build() == LinearExpression(((0, 1.5), (1, 6.0)), 4.0)

    builder.clear()
    assert builder.build() == LinearExpression()

    with pytest.raises(ValueError, match="constants must be finite"):
        LinearExpressionBuilder(constant=float("nan"))
    with pytest.raises(ValueError, match="coefficients must be finite"):
        builder.add_term(0, float("inf"))
    with pytest.raises(ValueError, match="scale must be finite"):
        builder.add_expression(LinearExpression(), factor=float("inf"))
    with pytest.raises(ValueError, match="constants must be finite"):
        builder.add_constant(float("nan"))


def test_variable_key_validation_and_display() -> None:
    scalar = VariableKey("auxiliary")
    indexed = VariableKey("allocation", (2, 3))

    assert str(scalar) == "auxiliary"
    assert str(indexed) == "allocation[2,3]"

    for block in ("", "   ", 3):
        with pytest.raises(ValueError, match="non-empty string"):
            VariableKey(block)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="coordinates must be integers"):
        VariableKey("x", (True,))
    with pytest.raises(TypeError, match="coordinates must be integers"):
        VariableKey("x", (1.5,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        VariableKey("x", (-1,))


def test_variable_index_is_an_immutable_ordered_bijection() -> None:
    keys = (
        VariableKey("allocation", (0, 0)),
        VariableKey("allocation", (0, 1)),
        VariableKey("payment", (0, 0)),
    )
    index = VariableIndex(keys)

    assert len(index) == 3
    assert tuple(index) == keys
    assert index[0] == keys[0]
    assert index[1:] == keys[1:]
    assert keys[1] in index
    assert index.keys == keys
    assert dict(index.positions) == {key: position for position, key in enumerate(keys)}
    assert index.blocks == ("allocation", "payment")
    assert index.position(keys[2]) == 2
    assert index.key(1) == keys[1]
    assert index.block_positions("allocation") == (0, 1)
    assert index.get(keys[0]) == 0
    assert index.get(VariableKey("missing"), 99) == 99
    assert repr(index) == f"VariableIndex({keys!r})"

    with pytest.raises(KeyError, match="unknown variable block"):
        index.block_positions("utility")
    with pytest.raises(TypeError, match="VariableKey"):
        VariableIndex(("allocation",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate variable key"):
        VariableIndex((keys[0], keys[0]))


def test_variable_index_builder_freezes_deterministically() -> None:
    builder = VariableIndexBuilder()
    first = builder.add("allocation", 0, 0)
    second = builder.add_key(VariableKey("payment", (0, 0)))
    rest = builder.add_block("slack", ((0,), (1,)))

    assert (first, second, rest) == (0, 1, (2, 3))
    assert len(builder) == 4
    assert builder.position(VariableKey("slack", (1,))) == 3
    with pytest.raises(ValueError, match="duplicate variable key"):
        builder.add("allocation", 0, 0)

    index = builder.freeze()
    assert index.blocks == ("allocation", "payment", "slack")
    with pytest.raises(RuntimeError, match="after the index has been frozen"):
        builder.add("late")
