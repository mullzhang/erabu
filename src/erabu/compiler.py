"""Translate bounded binary polynomial models to sparse mixed-integer programs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import ceil, floor, isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array

from ._nodes import (
    Aggregate,
    Constant,
    Node,
    PairSum,
    Rule,
    Scale,
    Square,
    Sum,
    Variance,
    degree,
    finite,
)
from .errors import ModelSizeError
from .result import ModelInfo


def support(node: Node) -> set[int]:
    match node:
        case Constant():
            return set()
        case Aggregate(terms):
            return {p for p, value in terms if value != 0}
        case Sum(children) | Variance(children):
            result: set[int] = set()
            for child in children:
                result.update(support(child))
            return result
        case Scale(child, factor):
            return support(child) if factor else set()
        case Square(child):
            return support(child)
        case PairSum(terms):
            return {p for i, j, v in terms if v != 0 for p in (i, j)}


def inspect_model(n: int, objective: Node, rules: Sequence[Rule]) -> ModelInfo:
    seen: set[int] = set()

    def bound(node: Node) -> int:
        if id(node) in seen:
            return 0
        seen.add(id(node))
        match node:
            case Constant() | Aggregate():
                return 0
            case PairSum(terms):
                return sum(v != 0 for _, _, v in terms)
            case Sum(children):
                return sum(bound(child) for child in children)
            case Scale(child, factor):
                return bound(child) if factor else 0
            case Square() | Variance():
                size = len(support(node))
                return size * (size - 1) // 2

    count = bound(objective)
    for rule in rules:
        count += bound(rule.comparison.left) + bound(rule.comparison.right)
    quadratic = degree(objective) == 2 or any(
        degree(rule.comparison.left) == 2 or degree(rule.comparison.right) == 2 for rule in rules
    )
    return ModelInfo(n, len(rules), count, quadratic)


@dataclass
class Polynomial:
    constant: float = 0.0
    linear: dict[int, float] = field(default_factory=dict)
    pairs: dict[tuple[int, int], float] = field(default_factory=dict)

    def add(self, other: Polynomial, factor: float = 1.0) -> None:
        self.constant = finite(self.constant + factor * other.constant)
        accumulate(self.linear, other.linear, factor)
        accumulate(self.pairs, other.pairs, factor)


def accumulate[K](destination: dict[K, float], source: dict[K, float], factor: float) -> None:
    for key, value in source.items():
        destination[key] = finite(destination.get(key, 0.0) + factor * value)
        if destination[key] == 0.0:
            del destination[key]


def squared(poly: Polynomial) -> Polynomial:
    result = Polynomial(finite(poly.constant * poly.constant))
    result.linear = {
        p: finite(v * v + 2 * poly.constant * v) for p, v in poly.linear.items() if v != 0
    }
    result.pairs = {
        (min(p, q), max(p, q)): finite(2 * vp * vq)
        for (p, vp), (q, vq) in combinations(poly.linear.items(), 2)
    }
    return result


class Compiler:
    def __init__(self) -> None:
        self.cache: dict[int, Polynomial] = {}

    def polynomial(self, node: Node) -> Polynomial:
        if id(node) not in self.cache:
            self.cache[id(node)] = self._polynomial(node)
        return self.cache[id(node)]

    def _polynomial(self, node: Node) -> Polynomial:
        match node:
            case Constant(value):
                return Polynomial(value)
            case Aggregate(terms):
                result = Polynomial()
                for p, value in terms:
                    result.linear[p] = finite(result.linear.get(p, 0.0) + value)
                result.linear = {p: v for p, v in result.linear.items() if v != 0}
                return result
            case PairSum(terms):
                return Polynomial(pairs={(p, q): v for p, q, v in terms if v != 0})
            case Sum(children):
                result = Polynomial()
                for child in children:
                    result.add(self.polynomial(child))
                return result
            case Scale(child, factor):
                result = Polynomial()
                if factor:
                    result.add(self.polynomial(child), factor)
                return result
            case Square(child):
                return squared(self.polynomial(child))
            case Variance(children):
                polys = [self.polynomial(child) for child in children]
                mean = Polynomial()
                for poly in polys:
                    mean.add(poly, 1 / len(polys))
                result = Polynomial()
                for poly in polys:
                    centered = Polynomial()
                    centered.add(poly)
                    centered.add(mean, -1)
                    result.add(squared(centered), 1 / len(polys))
                return result


@dataclass(frozen=True)
class CompiledModel:
    objective: NDArray[np.float64]
    offset: float
    sign: float
    integrality: NDArray[np.int32]
    matrix: Any
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    pair_count: int
    info: ModelInfo
    pairs: tuple[tuple[int, int], ...]


def compile_model(
    n: int, objective: Node, rules: Sequence[Rule], *, maximize: bool, max_pair_terms: int
) -> CompiledModel:
    info = inspect_model(n, objective, rules)
    if info.pair_terms_upper_bound > max_pair_terms:
        raise ModelSizeError(
            f"Estimated pair terms {info.pair_terms_upper_bound:,} exceed max_pair_terms="
            f"{max_pair_terms:,}. Reduce the model or explicitly increase the limit."
        )
    compiler = Compiler()
    cost = compiler.polynomial(objective)
    rows: list[tuple[Polynomial, float, float]] = []
    all_pairs = set(cost.pairs)
    for rule in rules:
        comparison = rule.comparison
        poly = Polynomial()
        poly.add(compiler.polynomial(comparison.left))
        poly.add(compiler.polynomial(comparison.right), -1)
        lower = -poly.constant if comparison.sense in (">=", "==") else -np.inf
        upper = -poly.constant if comparison.sense in ("<=", "==") else np.inf
        if all(v.is_integer() for v in (*poly.linear.values(), *poly.pairs.values())):
            # The variable part is integer-valued, even when the boundary is not.
            lower = float(ceil(lower)) if isfinite(lower) else lower
            upper = float(floor(upper)) if isfinite(upper) else upper
        rows.append((poly, lower, upper))
        all_pairs.update(poly.pairs)
    pairs = {pair: n + i for i, pair in enumerate(sorted(all_pairs))}
    size = n + len(pairs)
    sign = -1.0 if maximize else 1.0
    coefficients = np.zeros(size, dtype=np.float64)
    for p, value in cost.linear.items():
        coefficients[p] = sign * value
    for pair, value in cost.pairs.items():
        coefficients[pairs[pair]] = sign * value
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_row(terms: dict[int, float], lower: float, upper: float) -> None:
        index = len(lower_bounds)
        for col, value in terms.items():
            if value:
                row_indices.append(index)
                col_indices.append(col)
                values.append(value)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    for poly, lower, upper in rows:
        terms = dict(poly.linear)
        terms.update({pairs[pair]: value for pair, value in poly.pairs.items()})
        add_row(terms, lower, upper)
    for (p, q), y in pairs.items():
        add_row({y: 1.0, p: -1.0}, -np.inf, 0.0)
        add_row({y: 1.0, q: -1.0}, -np.inf, 0.0)
        add_row({y: 1.0, p: -1.0, q: -1.0}, -1.0, np.inf)
    matrix = coo_array(
        (
            np.asarray(values, dtype=np.float64),
            (np.asarray(row_indices, dtype=np.int32), np.asarray(col_indices, dtype=np.int32)),
        ),
        shape=(len(lower_bounds), size),
    ).tocsc()
    return CompiledModel(
        coefficients,
        cost.constant,
        sign,
        np.concatenate((np.ones(n, dtype=np.int32), np.zeros(len(pairs), dtype=np.int32))),
        matrix,
        np.asarray(lower_bounds),
        np.asarray(upper_bounds),
        len(pairs),
        info,
        tuple(pairs),
    )
