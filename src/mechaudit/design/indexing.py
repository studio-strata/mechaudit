"""Deterministic column indexing for compiled mechanism-design LPs.

The optimization layer never uses user-facing labels as array indices.  A
``VariableKey`` instead consists of a stable block name and integer
coordinates into the environment's canonical axes.  This keeps sparse-model
fingerprints and solver layouts independent of Python object hashes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import overload


@dataclass(frozen=True, order=True)
class VariableKey:
    """A stable identifier for one LP column.

    Parameters
    ----------
    block:
        A non-empty, human-readable block name such as ``"allocation"`` or
        ``"payment"``.
    coordinates:
        Integer positions in the canonical axes for that block.  A scalar
        auxiliary variable may use an empty coordinate tuple.
    """

    block: str
    coordinates: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.block, str) or not self.block.strip():
            raise ValueError("variable block must be a non-empty string")
        coordinates = tuple(self.coordinates)
        for coordinate in coordinates:
            if isinstance(coordinate, bool) or not isinstance(coordinate, int):
                raise TypeError("variable coordinates must be integers")
            if coordinate < 0:
                raise ValueError("variable coordinates must be nonnegative")
        object.__setattr__(self, "coordinates", coordinates)

    def __str__(self) -> str:
        if not self.coordinates:
            return self.block
        return "{}[{}]".format(self.block, ",".join(map(str, self.coordinates)))


KeyLike = VariableKey | tuple[str, tuple[int, ...]]


class VariableIndex(Sequence[VariableKey]):
    """An immutable bijection between :class:`VariableKey` objects and columns."""

    __slots__ = ("_blocks", "_keys", "_positions")

    def __init__(self, keys: Iterable[VariableKey]) -> None:
        canonical = tuple(keys)
        positions: dict[VariableKey, int] = {}
        blocks: dict[str, list[int]] = {}
        for position, key in enumerate(canonical):
            if not isinstance(key, VariableKey):
                raise TypeError("VariableIndex entries must be VariableKey objects")
            if key in positions:
                raise ValueError(f"duplicate variable key: {key}")
            positions[key] = position
            blocks.setdefault(key.block, []).append(position)

        self._keys = canonical
        self._positions = MappingProxyType(positions)
        self._blocks = MappingProxyType(
            {block: tuple(indices) for block, indices in blocks.items()}
        )

    @overload
    def __getitem__(self, item: int) -> VariableKey: ...

    @overload
    def __getitem__(self, item: slice) -> tuple[VariableKey, ...]: ...

    def __getitem__(self, item: int | slice) -> VariableKey | tuple[VariableKey, ...]:
        return self._keys[item]

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[VariableKey]:
        return iter(self._keys)

    def __contains__(self, item: object) -> bool:
        return item in self._positions

    @property
    def keys(self) -> tuple[VariableKey, ...]:
        """Return all keys in solver-column order."""

        return self._keys

    @property
    def positions(self) -> Mapping[VariableKey, int]:
        """Read-only key-to-column mapping."""

        return self._positions

    @property
    def blocks(self) -> tuple[str, ...]:
        """Block names in first-declaration order."""

        return tuple(self._blocks)

    def position(self, key: VariableKey) -> int:
        """Return the solver column for *key*."""

        return self._positions[key]

    def key(self, position: int) -> VariableKey:
        """Return the key stored at a solver column."""

        return self._keys[position]

    def block_positions(self, block: str) -> tuple[int, ...]:
        """Return all columns in *block*, preserving declaration order."""

        try:
            return self._blocks[block]
        except KeyError as exc:
            raise KeyError(f"unknown variable block: {block!r}") from exc

    def get(self, key: VariableKey, default: int | None = None) -> int | None:
        """Return a column position, or *default* when the key is absent."""

        return self._positions.get(key, default)

    def __repr__(self) -> str:
        return f"VariableIndex({self._keys!r})"


class VariableIndexBuilder:
    """Mutable construction helper that freezes into a :class:`VariableIndex`."""

    __slots__ = ("_frozen", "_keys", "_positions")

    def __init__(self) -> None:
        self._keys: list[VariableKey] = []
        self._positions: dict[VariableKey, int] = {}
        self._frozen = False

    def add(self, block: str, *coordinates: int) -> int:
        """Append a key and return its deterministic column position."""

        if self._frozen:
            raise RuntimeError("cannot add variables after the index has been frozen")
        key = VariableKey(block=block, coordinates=tuple(coordinates))
        if key in self._positions:
            raise ValueError(f"duplicate variable key: {key}")
        position = len(self._keys)
        self._keys.append(key)
        self._positions[key] = position
        return position

    def add_key(self, key: VariableKey) -> int:
        """Append an already constructed key."""

        return self.add(key.block, *key.coordinates)

    def add_block(self, block: str, coordinates: Iterable[tuple[int, ...]]) -> tuple[int, ...]:
        """Append an ordered collection of coordinates for one block."""

        return tuple(self.add(block, *tuple(coordinate)) for coordinate in coordinates)

    def position(self, key: VariableKey) -> int:
        """Return a position already assigned by this builder."""

        return self._positions[key]

    def freeze(self) -> VariableIndex:
        """Freeze the current layout and prevent subsequent mutation."""

        self._frozen = True
        return VariableIndex(self._keys)

    def __len__(self) -> int:
        return len(self._keys)
