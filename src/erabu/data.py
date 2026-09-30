"""Validated snapshots of candidate tables and their relations."""

from collections.abc import Iterable
from numbers import Integral, Real

import numpy as np
import pandas as pd

type Label = str | int


def number(value: object, *, context: str = "Value") -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{context} must be a finite real number, got {value!r}.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be finite, got {value!r}.")
    return result


def labels(values: Iterable[object], *, context: str, unique: bool = True) -> tuple[Label, ...]:
    if isinstance(values, (str, bytes, dict, set, frozenset)):
        raise TypeError(f"{context} must be an ordered iterable of string or integer IDs.")
    result: list[Label] = []
    kind: type | None = None
    seen: set[Label] = set()
    for value in values:
        if isinstance(value, str):
            item: Label = value
        elif isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
            item = int(value)
        else:
            raise ValueError(f"{context} contains an invalid ID: {value!r}.")
        if kind is not None and type(item) is not kind:
            raise ValueError(f"{context} mixes string and integer IDs.")
        kind = type(item)
        if unique and item in seen:
            raise ValueError(f"{context} contains duplicate ID {item!r}.")
        seen.add(item)
        result.append(item)
    return tuple(result)


def frame(table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if not table.columns.is_unique or any(not isinstance(c, str) for c in table.columns):
        raise ValueError("Column names must be unique strings.")
    for column in table.columns:
        if any(not pd.api.types.is_scalar(v) for v in table[column]):
            raise ValueError(
                f"Column {column!r} contains mutable or nested values; keep them by ID."
            )
    return table.copy(deep=True)


def column(table: pd.DataFrame, name: str) -> pd.Series:
    if name not in table.columns:
        raise ValueError(f"Unknown column {name!r}.")
    return table[name]


def numeric_column(
    table: pd.DataFrame, name: str, *, nonnegative: bool = False
) -> tuple[float, ...]:
    result = tuple(number(v, context=f"Column {name!r}") for v in column(table, name))
    if nonnegative and any(v < 0 for v in result):
        raise ValueError(f"Column {name!r} must contain nonnegative quantities.")
    return result


class CandidateData:
    def __init__(self, table: pd.DataFrame, key: str) -> None:
        self.table = frame(table)
        self.ids = labels(column(self.table, key), context=f"Candidate key {key!r}")
        self.index = {label: i for i, label in enumerate(self.ids)}

    def positions(self, values: Iterable[object], *, unique: bool = False) -> tuple[int, ...]:
        ids = labels(values, context="Candidate references", unique=unique)
        unknown = [item for item in ids if item not in self.index]
        if unknown:
            raise ValueError(f"Unknown candidate IDs: {unknown!r}.")
        return tuple(self.index[item] for item in ids)

    def pairs(self, table: pd.DataFrame, left: str, right: str) -> tuple[tuple[int, int], ...]:
        left_ids = self.positions(column(table, left))
        right_ids = self.positions(column(table, right))
        result: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for p, q in zip(left_ids, right_ids, strict=True):
            if p == q:
                raise ValueError(f"Self-pair for candidate {self.ids[p]!r} is not allowed.")
            pair = (min(p, q), max(p, q))
            if pair in seen:
                raise ValueError("Duplicate candidate pair, including reversed pairs.")
            seen.add(pair)
            result.append(pair)
        return tuple(result)
