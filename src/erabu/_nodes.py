"""Expression definitions shared by compilation and direct evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from .data import Label
from .errors import NumericalError


def finite(value: float) -> float:
    if not isfinite(value):
        raise NumericalError("Expression arithmetic produced a non-finite value.")
    return value


@dataclass(frozen=True)
class Constant:
    value: float


@dataclass(frozen=True)
class Aggregate:
    terms: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class Sum:
    children: tuple[Node, ...]


@dataclass(frozen=True)
class Scale:
    child: Node
    factor: float


@dataclass(frozen=True)
class Square:
    child: Node


@dataclass(frozen=True)
class Variance:
    children: tuple[Node, ...]


@dataclass(frozen=True)
class PairSum:
    terms: tuple[tuple[int, int, float], ...]


type Node = Constant | Aggregate | Sum | Scale | Square | Variance | PairSum
type Sense = Literal["<=", ">=", "=="]


@dataclass(frozen=True)
class Comparison:
    left: Node
    right: Node
    sense: Sense


@dataclass(frozen=True)
class Rule:
    name: str
    target: Label | tuple[Label, Label] | None
    comparison: Comparison


def degree(node: Node) -> int:
    match node:
        case Constant():
            return 0
        case Aggregate(terms):
            return int(any(value != 0 for _, value in terms))
        case PairSum(terms):
            return 2 if any(value != 0 for _, _, value in terms) else 0
        case Sum(children):
            return max((degree(child) for child in children), default=0)
        case Scale(child, factor):
            return degree(child) if factor else 0
        case Square(child):
            return 2 * degree(child)
        case Variance(children):
            return 2 * max((degree(child) for child in children), default=0)


def integral(node: Node) -> bool:
    match node:
        case Constant(value):
            return value.is_integer()
        case Aggregate(terms):
            return all(value.is_integer() for _, value in terms)
        case PairSum(terms):
            return all(value.is_integer() for _, _, value in terms)
        case Sum(children):
            return all(integral(child) for child in children)
        case Scale(child, factor):
            return factor.is_integer() and integral(child)
        case Square(child):
            return integral(child)
        case Variance():
            return False
