"""Symbolic aggregates over a selection of candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import overload

import pandas as pd

from ._nodes import (
    Aggregate,
    Comparison,
    Constant,
    Node,
    PairSum,
    Scale,
    Sense,
    Square,
    Sum,
    Variance,
    degree,
)
from .data import CandidateData, Label, frame, labels, number, numeric_column
from .data import column as get_column


def same_problem(left: object, right: object) -> None:
    if left is not right:
        raise ValueError("Cannot mix expressions from different problems.")


def no_truth() -> bool:
    raise TypeError(
        "Symbolic expressions have no truth value; register each comparison separately."
    )


@dataclass(frozen=True)
class Condition:
    """A scalar or labeled set of comparisons, to be registered with Problem.require."""

    _owner: object
    _rows: tuple[tuple[Label | tuple[Label, Label] | None, Comparison], ...]

    def __bool__(self) -> bool:
        return no_truth()


class Scalar:
    """A scalar expression in the unknown selection."""

    def __init__(self, owner: object, node: Node) -> None:
        self._owner = owner
        self._node = node

    def _coerce(self, other: object) -> Node:
        if isinstance(other, Scalar):
            same_problem(self._owner, other._owner)
            return other._node
        return Constant(number(other))

    @overload
    def __add__(self, other: Vector) -> Vector: ...

    @overload
    def __add__(self, other: Scalar | float) -> Scalar: ...

    def __add__(self, other: object) -> Scalar | Vector:
        if isinstance(other, Vector):
            return other + self
        return Scalar(self._owner, Sum((self._node, self._coerce(other))))

    def __radd__(self, other: float) -> Scalar:
        return self + other

    @overload
    def __sub__(self, other: Vector) -> Vector: ...

    @overload
    def __sub__(self, other: Scalar | float) -> Scalar: ...

    def __sub__(self, other: object) -> Scalar | Vector:
        if isinstance(other, Vector):
            return -other + self
        return Scalar(self._owner, Sum((self._node, Scale(self._coerce(other), -1.0))))

    def __rsub__(self, other: float) -> Scalar:
        return -self + other

    def __neg__(self) -> Scalar:
        return self * -1

    def __mul__(self, factor: float) -> Scalar:
        return Scalar(self._owner, Scale(self._node, number(factor, context="Multiplier")))

    def __rmul__(self, factor: float) -> Scalar:
        return self * factor

    def __truediv__(self, divisor: float) -> Scalar:
        return self * (1 / number(divisor, context="Divisor"))

    def square(self) -> Scalar:
        """Square an affine expression; higher-degree expressions are unsupported."""
        if degree(self._node) > 1:
            raise ValueError("Only affine expressions can be squared.")
        return Scalar(self._owner, Square(self._node))

    def _compare(self, other: object, sense: Sense) -> Condition:
        if isinstance(other, Vector):
            reverse: dict[Sense, Sense] = {"<=": ">=", ">=": "<=", "==": "=="}
            return other._compare(self, reverse[sense])
        return Condition(self._owner, ((None, Comparison(self._node, self._coerce(other), sense)),))

    def __le__(self, other: object) -> Condition:
        return self._compare(other, "<=")

    def __ge__(self, other: object) -> Condition:
        return self._compare(other, ">=")

    def __eq__(self, other: object) -> Condition:  # type: ignore[override]
        return self._compare(other, "==")

    def __bool__(self) -> bool:
        return no_truth()


class Vector:
    """Expressions over an explicit, fixed set of labels."""

    def __init__(self, owner: object, domain: tuple[Label, ...], nodes: tuple[Node, ...]) -> None:
        self._owner = owner
        self._labels = domain
        self._nodes = nodes

    @property
    def labels(self) -> tuple[Label, ...]:
        return self._labels

    def _aligned(self, other: object) -> tuple[Node, ...]:
        if isinstance(other, Vector):
            same_problem(self._owner, other._owner)
            if set(self._labels) != set(other._labels):
                raise ValueError("Vector labels must match exactly.")
            lookup = dict(zip(other._labels, other._nodes, strict=True))
            return tuple(lookup[label] for label in self._labels)
        if isinstance(other, Mapping):
            keys = labels(list(other), context="Constant vector labels")
            if set(keys) != set(self._labels):
                raise ValueError("Constant vector labels must match exactly.")
            return tuple(Constant(number(other[label])) for label in self._labels)
        if isinstance(other, Scalar):
            same_problem(self._owner, other._owner)
            node = other._node
        else:
            node = Constant(number(other))
        return (node,) * len(self._labels)

    def __add__(self, other: object) -> Vector:
        return Vector(
            self._owner,
            self._labels,
            tuple(Sum((a, b)) for a, b in zip(self._nodes, self._aligned(other), strict=True)),
        )

    def __radd__(self, other: object) -> Vector:
        return self + other

    def __sub__(self, other: object) -> Vector:
        return Vector(
            self._owner,
            self._labels,
            tuple(
                Sum((a, Scale(b, -1.0)))
                for a, b in zip(self._nodes, self._aligned(other), strict=True)
            ),
        )

    def __rsub__(self, other: object) -> Vector:
        return -self + other

    def __neg__(self) -> Vector:
        return self * -1

    def __mul__(self, factor: float) -> Vector:
        value = number(factor, context="Multiplier")
        return Vector(self._owner, self._labels, tuple(Scale(n, value) for n in self._nodes))

    def __rmul__(self, factor: float) -> Vector:
        return self * factor

    def __truediv__(self, divisor: float) -> Vector:
        return self * (1 / number(divisor, context="Divisor"))

    def sum(self) -> Scalar:
        return Scalar(self._owner, Sum(self._nodes))

    def mean(self) -> Scalar:
        if not self._nodes:
            raise ValueError("The mean requires a nonempty target universe.")
        return self.sum() / len(self._nodes)

    def square(self) -> Vector:
        return Vector(
            self._owner,
            self._labels,
            tuple(Scalar(self._owner, n).square()._node for n in self._nodes),
        )

    def variance(self, *, ddof: int = 0) -> Scalar:
        """Population variance across these fixed labels, including zero totals."""
        if isinstance(ddof, bool) or ddof != 0:
            raise ValueError("Only population variance (ddof=0) is supported.")
        if not self._nodes:
            raise ValueError("Variance requires a nonempty target universe.")
        if any(degree(n) > 1 for n in self._nodes):
            raise ValueError("Variance requires affine expressions.")
        return Scalar(self._owner, Variance(self._nodes))

    def _compare(self, other: object, sense: Sense) -> Condition:
        return Condition(
            self._owner,
            tuple(
                (label, Comparison(a, b, sense))
                for label, a, b in zip(self._labels, self._nodes, self._aligned(other), strict=True)
            ),
        )

    def __le__(self, other: object) -> Condition:
        return self._compare(other, "<=")

    def __ge__(self, other: object) -> Condition:
        return self._compare(other, ">=")

    def __eq__(self, other: object) -> Condition:  # type: ignore[override]
        return self._compare(other, "==")

    def __bool__(self) -> bool:
        return no_truth()


class GroupedSelection:
    """One-to-many aggregate rows indexed by the complete target universe."""

    def __init__(
        self,
        owner: object,
        table: pd.DataFrame,
        positions: tuple[int, ...],
        items: tuple[Label, ...],
        universe: Iterable[object],
        *,
        quantities: bool,
    ) -> None:
        self._owner = owner
        self._table = table
        self._positions = positions
        self._domain = labels(universe, context="Target universe")
        self._groups: dict[Label, list[int]] = {label: [] for label in self._domain}
        for row, item in enumerate(items):
            if item not in self._groups:
                raise ValueError(f"Target {item!r} is outside the explicit universe.")
            self._groups[item].append(row)
        self._quantities = quantities

    def _aggregate(self, values: tuple[float, ...]) -> Vector:
        return Vector(
            self._owner,
            self._domain,
            tuple(
                Aggregate(tuple((self._positions[row], values[row]) for row in self._groups[label]))
                for label in self._domain
            ),
        )

    def count(self) -> Vector:
        return self._aggregate((1.0,) * len(self._positions))

    def sum(self, column: str) -> Vector:
        return self._aggregate(numeric_column(self._table, column, nonnegative=self._quantities))


class Selection:
    """Entry point for aggregates over the yet-to-be-selected candidates."""

    def __init__(self, data: CandidateData, owner: object) -> None:
        self._data = data
        self._owner = owner

    def count(self) -> Scalar:
        return Scalar(self._owner, Aggregate(tuple((i, 1.0) for i in range(len(self._data.ids)))))

    @overload
    def sum(self, columns: str) -> Scalar: ...

    @overload
    def sum(self, columns: list[str] | tuple[str, ...]) -> Vector: ...

    def sum(self, columns: str | list[str] | tuple[str, ...]) -> Scalar | Vector:
        if isinstance(columns, str):
            values = numeric_column(self._data.table, columns)
            return Scalar(self._owner, Aggregate(tuple(enumerate(values))))
        names = labels(columns, context="Aggregate columns")
        if not names or any(not isinstance(name, str) for name in names):
            raise ValueError("Aggregate columns must be a nonempty list of unique column names.")
        nodes = tuple(
            Aggregate(tuple(enumerate(numeric_column(self._data.table, str(name)))))
            for name in names
        )
        return Vector(self._owner, names, nodes)

    def groupby(self, column: str, *, universe: Iterable[object]) -> GroupedSelection:
        return GroupedSelection(
            self._owner,
            self._data.table,
            tuple(range(len(self._data.ids))),
            labels(get_column(self._data.table, column), context=f"Group {column!r}", unique=False),
            universe,
            quantities=False,
        )

    def related(
        self,
        table: pd.DataFrame,
        *,
        pattern: str,
        item: str,
        universe: Iterable[object],
    ) -> GroupedSelection:
        snapshot = frame(table)
        positions = self._data.positions(get_column(snapshot, pattern))
        items = labels(get_column(snapshot, item), context="Relation targets", unique=False)
        if len(set(zip(positions, items, strict=True))) != len(positions):
            raise ValueError("Duplicate candidate-target relation; aggregate quantities first.")
        return GroupedSelection(self._owner, snapshot, positions, items, universe, quantities=True)

    def pair_sum(self, table: pd.DataFrame, *, left: str, right: str, value: str) -> Scalar:
        snapshot = frame(table)
        pairs = self._data.pairs(snapshot, left, right)
        values = numeric_column(snapshot, value)
        return Scalar(
            self._owner,
            PairSum(tuple((p, q, v) for (p, q), v in zip(pairs, values, strict=True))),
        )
