"""Small, solver-independent linear-expression primitives.

Expressions are canonicalized as sorted ``(column, coefficient)`` pairs.
Keeping this layer independent of SciPy makes it convenient for requirement
compilers to assemble rows while :mod:`mechaudit.design.sparse_model` remains
responsible for bulk sparse storage.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np

Number = int | float
Term = tuple[int, float]


def _validate_column(column: int) -> None:
    if isinstance(column, bool) or not isinstance(column, int):
        raise TypeError("linear-expression column indices must be integers")
    if column < 0:
        raise ValueError("linear-expression column indices must be nonnegative")


@dataclass(frozen=True)
class LinearExpression:
    """An immutable sparse affine expression.

    Duplicate columns are summed with :func:`math.fsum`, exact zeros are
    removed, and terms are sorted by column.  Tiny nonzero coefficients are
    retained; dropping them is a numerical-policy decision, not
    an expression-building side effect.
    """

    terms: tuple[Term, ...] = ()
    constant: float = 0.0

    def __post_init__(self) -> None:
        grouped: dict[int, list[float]] = {}
        for column, coefficient in tuple(self.terms):
            _validate_column(column)
            value = float(coefficient)
            if not math.isfinite(value):
                raise ValueError("linear-expression coefficients must be finite")
            grouped.setdefault(column, []).append(value)

        canonical = []
        for column in sorted(grouped):
            coefficient = math.fsum(grouped[column])
            if coefficient != 0.0:
                canonical.append((column, coefficient))

        constant = float(self.constant)
        if not math.isfinite(constant):
            raise ValueError("linear-expression constants must be finite")
        if constant == 0.0:
            constant = 0.0  # Canonicalize negative zero for stable fingerprints.

        object.__setattr__(self, "terms", tuple(canonical))
        object.__setattr__(self, "constant", constant)

    @classmethod
    def from_mapping(
        cls, coefficients: Mapping[int, Number], *, constant: Number = 0.0
    ) -> LinearExpression:
        """Create an expression from a column-to-coefficient mapping."""

        return cls(
            tuple((column, float(coefficient)) for column, coefficient in coefficients.items()),
            float(constant),
        )

    @property
    def indices(self) -> tuple[int, ...]:
        """Column indices in ascending order."""

        return tuple(column for column, _ in self.terms)

    @property
    def coefficients(self) -> tuple[float, ...]:
        """Coefficients corresponding to :attr:`indices`."""

        return tuple(coefficient for _, coefficient in self.terms)

    def coefficient(self, column: int) -> float:
        """Return one coefficient, or zero when the column is absent."""

        _validate_column(column)
        for candidate, value in self.terms:
            if candidate == column:
                return value
            if candidate > column:
                break
        return 0.0

    def evaluate(self, values: Iterable[Number]) -> float:
        """Evaluate the affine expression at a dense vector."""

        vector = np.asarray(tuple(values), dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("linear-expression values must be one-dimensional")
        if self.terms and self.terms[-1][0] >= vector.size:
            raise ValueError("linear-expression column is outside the supplied vector")
        if not np.all(np.isfinite(vector)):
            raise ValueError("linear-expression values must be finite")
        return self.constant + math.fsum(
            coefficient * float(vector[column]) for column, coefficient in self.terms
        )

    def scaled(self, factor: Number) -> LinearExpression:
        """Return this expression multiplied by *factor*."""

        scale = float(factor)
        if not math.isfinite(scale):
            raise ValueError("linear-expression scale must be finite")
        return LinearExpression(
            tuple((column, scale * value) for column, value in self.terms),
            scale * self.constant,
        )

    def without_constant(self) -> LinearExpression:
        """Return the linear portion of this affine expression."""

        if self.constant == 0.0:
            return self
        return LinearExpression(self.terms)

    def __iter__(self) -> Iterator[Term]:
        return iter(self.terms)

    def __neg__(self) -> LinearExpression:
        return self.scaled(-1.0)

    def __add__(self, other: object) -> LinearExpression:
        if isinstance(other, LinearExpression):
            return LinearExpression(self.terms + other.terms, self.constant + other.constant)
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return LinearExpression(self.terms, self.constant + float(other))
        return NotImplemented

    def __radd__(self, other: object) -> LinearExpression:
        return self.__add__(other)

    def __sub__(self, other: object) -> LinearExpression:
        if isinstance(other, LinearExpression):
            return self + (-other)
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return LinearExpression(self.terms, self.constant - float(other))
        return NotImplemented

    def __rsub__(self, other: object) -> LinearExpression:
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return (-self) + float(other)
        return NotImplemented

    def __mul__(self, factor: object) -> LinearExpression:
        if isinstance(factor, (int, float)) and not isinstance(factor, bool):
            return self.scaled(float(factor))
        return NotImplemented

    def __rmul__(self, factor: object) -> LinearExpression:
        return self.__mul__(factor)


# Short alias used frequently by requirement compilers.
LinearExpr = LinearExpression


class LinearExpressionBuilder:
    """Mutable accumulator for a single sparse affine expression."""

    __slots__ = ("_constant", "_terms")

    def __init__(self, *, constant: Number = 0.0) -> None:
        value = float(constant)
        if not math.isfinite(value):
            raise ValueError("linear-expression constants must be finite")
        self._constant = value
        self._terms: dict[int, list[float]] = {}

    def add_term(self, column: int, coefficient: Number) -> LinearExpressionBuilder:
        """Accumulate one coefficient and return ``self`` for chaining."""

        _validate_column(column)
        value = float(coefficient)
        if not math.isfinite(value):
            raise ValueError("linear-expression coefficients must be finite")
        self._terms.setdefault(column, []).append(value)
        return self

    def add_expression(
        self, expression: LinearExpression, *, factor: Number = 1.0
    ) -> LinearExpressionBuilder:
        """Accumulate a scaled expression."""

        scale = float(factor)
        if not math.isfinite(scale):
            raise ValueError("linear-expression scale must be finite")
        for column, coefficient in expression:
            self.add_term(column, scale * coefficient)
        self._constant = math.fsum((self._constant, scale * expression.constant))
        return self

    def add_constant(self, value: Number) -> LinearExpressionBuilder:
        """Accumulate a constant."""

        constant = float(value)
        if not math.isfinite(constant):
            raise ValueError("linear-expression constants must be finite")
        self._constant = math.fsum((self._constant, constant))
        return self

    def build(self) -> LinearExpression:
        """Return a canonical immutable expression."""

        terms = tuple((column, math.fsum(values)) for column, values in self._terms.items())
        return LinearExpression(terms, self._constant)

    def clear(self) -> None:
        """Reset the builder to zero."""

        self._terms.clear()
        self._constant = 0.0
