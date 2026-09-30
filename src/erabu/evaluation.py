"""Evaluate original expressions without using compiled model coefficients."""

from collections.abc import Mapping, Sequence
from math import fsum
from types import MappingProxyType

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
    integral,
)
from .errors import NumericalError
from .expressions import Scalar, Vector
from .result import ConstraintCheck, Evaluation, MetricValue, Tolerances


class Evaluator:
    def __init__(self, selected: set[int]) -> None:
        self.selected = selected
        self._cache: dict[int, float] = {}

    def value(self, node: Node) -> float:
        if id(node) not in self._cache:
            try:
                self._cache[id(node)] = finite(self._value(node))
            except OverflowError as exc:
                raise NumericalError("Expression evaluation overflowed.") from exc
        return self._cache[id(node)]

    def _value(self, node: Node) -> float:
        match node:
            case Constant(value):
                return value
            case Aggregate(terms):
                return fsum(value for p, value in terms if p in self.selected)
            case PairSum(terms):
                return fsum(v for p, q, v in terms if p in self.selected and q in self.selected)
            case Sum(children):
                return fsum(self.value(child) for child in children)
            case Scale(child, factor):
                return self.value(child) * factor
            case Square(child):
                value = self.value(child)
                return value * value
            case Variance(children):
                values = [self.value(child) for child in children]
                mean = fsum(values) / len(values)
                return fsum((v - mean) * (v - mean) for v in values) / len(values)


def evaluate(
    selected: set[int],
    objective: Node,
    rules: Sequence[Rule],
    metrics: Mapping[str, Scalar | Vector],
    tolerances: Tolerances,
) -> Evaluation:
    evaluator = Evaluator(selected)
    checks: list[ConstraintCheck] = []
    for rule in rules:
        comparison = rule.comparison
        left, right = evaluator.value(comparison.left), evaluator.value(comparison.right)
        lower = right if comparison.sense in (">=", "==") else None
        upper = right if comparison.sense in ("<=", "==") else None
        violation = finite(
            max(0.0, 0.0 if lower is None else lower - left, 0.0 if upper is None else left - upper)
        )
        exact = (
            integral(comparison.left)
            and (integral(comparison.right) or degree(comparison.right) == 0)
        ) or (integral(comparison.right) and degree(comparison.left) == 0)
        tolerance = 0.0 if exact else finite(tolerances.allowance(left, right))
        checks.append(
            ConstraintCheck(
                rule.name,
                rule.target,
                left,
                lower,
                upper,
                violation,
                tolerance,
                violation <= tolerance,
            )
        )
    values: dict[str, MetricValue] = {}
    for name, expression in metrics.items():
        if isinstance(expression, Scalar):
            values[name] = evaluator.value(expression._node)
        else:
            values[name] = MappingProxyType(
                {
                    label: evaluator.value(node)
                    for label, node in zip(expression._labels, expression._nodes, strict=True)
                }
            )
    return Evaluation(
        all(check.satisfied for check in checks),
        evaluator.value(objective),
        MappingProxyType(values),
        tuple(checks),
    )
