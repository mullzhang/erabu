"""Build, solve, and independently evaluate a candidate-selection problem."""

from collections.abc import Iterable
from numbers import Integral
from time import perf_counter
from types import MappingProxyType

import numpy as np
import pandas as pd

from ._nodes import Aggregate, Comparison, Constant, Node, Rule, finite
from .compiler import compile_model, inspect_model
from .data import CandidateData, frame, number
from .errors import NumericalError
from .evaluation import evaluate
from .expressions import Condition, Scalar, Selection, Vector, same_problem
from .result import (
    Bundle,
    Evaluation,
    ModelInfo,
    Result,
    Status,
    Termination,
    Tolerances,
    Verification,
)
from .solver import run_solver

DEFAULT_TOLERANCES = Tolerances()


def nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return int(value)


def valid_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Names must be nonempty strings.")


class Problem:
    """Select a verified bundle from a fixed snapshot of a candidate DataFrame."""

    def __init__(self, patterns: pd.DataFrame, *, key: str) -> None:
        started = perf_counter()
        self._data = CandidateData(patterns, key)
        self._input_seconds = perf_counter() - started
        self._owner = object()
        self._selected = Selection(self._data, self._owner)
        self._rules: list[Rule] = []
        self._rule_names: set[str] = set()
        self._metrics: dict[str, Scalar | Vector] = {}
        self._objective_node: Node | None = None
        self._maximize = False

    @property
    def selected(self) -> Selection:
        return self._selected

    def require(self, condition: Condition, *, name: str) -> None:
        """Register all comparisons in a named scalar or vector condition."""
        valid_name(name)
        if name in self._rule_names:
            raise ValueError(f"Duplicate condition name {name!r}.")
        if not isinstance(condition, Condition):
            raise TypeError("require() expects a symbolic comparison, not a boolean.")
        same_problem(self._owner, condition._owner)
        self._rules.extend(Rule(name, target, comparison) for target, comparison in condition._rows)
        self._rule_names.add(name)

    def cover(
        self,
        *,
        membership: pd.DataFrame,
        pattern: str,
        item: str,
        universe: Iterable[object],
        name: str,
        exactly: int | None = None,
        at_least: int | None = None,
        at_most: int | None = None,
    ) -> None:
        """Constrain coverage counts, preserving targets without candidates."""
        if exactly is not None and (at_least is not None or at_most is not None):
            raise ValueError("exactly cannot be combined with at_least or at_most.")
        if exactly is None and at_least is None and at_most is None:
            raise ValueError("Specify at least one coverage bound.")
        for key, value in (("exactly", exactly), ("at_least", at_least), ("at_most", at_most)):
            if value is not None:
                nonnegative_integer(value, name=key)
        if at_least is not None and at_most is not None and at_least > at_most:
            raise ValueError("Coverage lower bound exceeds upper bound.")
        counts = self.selected.related(
            membership, pattern=pattern, item=item, universe=universe
        ).count()
        if exactly is not None:
            condition = counts == exactly
        else:
            conditions: list[Condition] = []
            if at_least is not None:
                conditions.append(counts >= at_least)
            if at_most is not None:
                conditions.append(counts <= at_most)
            condition = Condition(self._owner, tuple(row for c in conditions for row in c._rows))
        self.require(condition, name=name)

    def forbid_pairs(self, table: pd.DataFrame, *, left: str, right: str, name: str) -> None:
        """Disallow selecting both members of each undirected candidate pair."""
        pairs = self._data.pairs(frame(table), left, right)
        self.require(
            Condition(
                self._owner,
                tuple(
                    (
                        (self._data.ids[p], self._data.ids[q]),
                        Comparison(Aggregate(((p, 1.0), (q, 1.0))), Constant(1.0), "<="),
                    )
                    for p, q in pairs
                ),
            ),
            name=name,
        )

    def _set_objective(self, expression: Scalar, *, maximize: bool) -> None:
        if self._objective_node is not None:
            raise ValueError("An objective is already registered.")
        if not isinstance(expression, Scalar):
            raise TypeError("The objective must be a scalar selection expression.")
        same_problem(self._owner, expression._owner)
        self._objective_node = expression._node
        self._maximize = maximize

    def minimize(self, expression: Scalar) -> None:
        self._set_objective(expression, maximize=False)

    def maximize(self, expression: Scalar) -> None:
        self._set_objective(expression, maximize=True)

    def metric(self, name: str, expression: Scalar | Vector) -> None:
        valid_name(name)
        if name in self._metrics:
            raise ValueError(f"Duplicate metric name {name!r}.")
        if not isinstance(expression, (Scalar, Vector)):
            raise TypeError("A metric must be a scalar or vector selection expression.")
        same_problem(self._owner, expression._owner)
        self._metrics[name] = expression

    def _objective(self) -> Node:
        if self._objective_node is None:
            raise ValueError("Register one objective before evaluating or solving.")
        return self._objective_node

    def inspect(self) -> ModelInfo:
        """Estimate model size without expanding binary products."""
        return inspect_model(len(self._data.ids), self._objective(), self._rules)

    def evaluate(
        self, pattern_ids: Iterable[object], *, tolerances: Tolerances = DEFAULT_TOLERANCES
    ) -> Evaluation:
        """Evaluate the original expressions; make no optimality claim."""
        positions = set(self._data.positions(pattern_ids, unique=True))
        return evaluate(positions, self._objective(), self._rules, self._metrics, tolerances)

    def solve(
        self,
        *,
        time_limit: float | None = None,
        node_limit: int | None = None,
        max_pair_terms: int = 100_000,
        tolerances: Tolerances = DEFAULT_TOLERANCES,
    ) -> Result:
        """Solve using HiGHS, then verify the selection against the original expressions.

        time_limit applies only to the solver. Compilation and verification are
        timed separately. max_pair_terms limits a conservative expansion estimate.
        """
        max_pair_terms = nonnegative_integer(max_pair_terms, name="max_pair_terms")
        if node_limit is not None:
            node_limit = nonnegative_integer(node_limit, name="node_limit")
        if time_limit is not None:
            time_limit = number(time_limit, context="time_limit")
            if time_limit < 0:
                raise ValueError("time_limit must be nonnegative.")
        objective = self._objective()
        started = perf_counter()
        n = len(self._data.ids)
        model = compile_model(
            n, objective, self._rules, maximize=self._maximize, max_pair_terms=max_pair_terms
        )
        diagnostics: dict[str, object] = {
            "input_seconds": self._input_seconds,
            "model": model.info,
            "pair_count": model.pair_count,
            "variable_count": len(model.objective),
            "constraint_count": len(model.lower),
            "compile_seconds": perf_counter() - started,
            "solve_seconds": 0.0,
            "verify_seconds": 0.0,
            "tolerances": tolerances,
        }

        def finish(
            status: Status,
            termination: Termination,
            verification: Verification,
            evaluation: Evaluation | None = None,
            selected: tuple[int, ...] = (),
            bound: float | None = None,
        ) -> Result:
            bundle = None
            if evaluation is not None:
                bundle = Bundle(
                    tuple(self._data.ids[p] for p in selected),
                    self._data.table.iloc[list(selected)].copy(deep=True),
                )
            return Result(
                status,
                termination,
                verification,
                bundle,
                None if evaluation is None else evaluation.objective_value,
                None if evaluation is None else evaluation.metrics,
                None if evaluation is None else evaluation.constraint_report,
                bound,
                None
                if evaluation is None or bound is None
                else finite(abs(evaluation.objective_value - bound)),
                MappingProxyType(dict(diagnostics)),
            )

        if n == 0:
            started = perf_counter()
            evaluation = evaluate(set(), objective, self._rules, self._metrics, tolerances)
            diagnostics["verify_seconds"] = perf_counter() - started
            diagnostics["message"] = "The unique empty selection was evaluated directly."
            if evaluation.feasible:
                return finish(
                    Status.OPTIMAL,
                    Termination.COMPLETED,
                    Verification.PASSED,
                    evaluation,
                    bound=evaluation.objective_value,
                )
            diagnostics["rejected_evaluation"] = evaluation
            return finish(Status.INFEASIBLE, Termination.COMPLETED, Verification.FAILED)

        started = perf_counter()
        try:
            raw = run_solver(model, time_limit=time_limit, node_limit=node_limit)
        except (RuntimeError, ValueError) as exc:
            diagnostics["solve_seconds"] = perf_counter() - started
            diagnostics["message"] = f"Solver raised {type(exc).__name__}: {exc}"
            return finish(Status.ERROR, Termination.SOLVER_ERROR, Verification.NOT_RUN)
        diagnostics.update(
            {
                "solve_seconds": perf_counter() - started,
                "solver_status": raw.status,
                "message": raw.message,
                "solver_objective": raw.objective,
                "solver_gap": raw.gap,
            }
        )
        termination = Termination.LIMIT if raw.status == 1 else Termination.COMPLETED
        if raw.status not in (0, 1, 2) or (raw.status == 2 and raw.x is not None):
            return finish(Status.ERROR, Termination.SOLVER_ERROR, Verification.NOT_RUN)
        if raw.status == 2:
            return finish(Status.INFEASIBLE, termination, Verification.NOT_RUN)
        if raw.x is None:
            if raw.status == 0:
                diagnostics["verification_error"] = "Solver reported optimality without a solution."
                return finish(Status.ERROR, Termination.SOLVER_ERROR, Verification.NOT_RUN)
            bound = None if raw.bound is None else model.offset + model.sign * raw.bound
            if bound is not None and not np.isfinite(bound):
                diagnostics["verification_error"] = "Solver returned a non-finite bound."
                return finish(Status.ERROR, Termination.SOLVER_ERROR, Verification.NOT_RUN)
            return finish(Status.NO_SOLUTION, termination, Verification.NOT_RUN, bound=bound)

        started = perf_counter()
        try:
            x = np.asarray(raw.x, dtype=float)
            if x.shape != model.objective.shape or not np.all(np.isfinite(x)):
                raise NumericalError("Solver returned a malformed or non-finite solution.")
            rounded = np.rint(x[:n])
            if (
                np.any(np.abs(x[:n] - rounded) > tolerances.integrality)
                or np.any(rounded < 0)
                or np.any(rounded > 1)
            ):
                raise NumericalError(
                    "Selection variables are not within tolerance of binary values."
                )
            expected = np.array([rounded[p] * rounded[q] for p, q in model.pairs])
            if np.any(np.abs(x[n:] - expected) > tolerances.absolute + tolerances.relative):
                raise NumericalError("Auxiliary variables do not match binary products.")
            selected = tuple(int(p) for p in np.flatnonzero(rounded))
            evaluation = evaluate(set(selected), objective, self._rules, self._metrics, tolerances)
            if not evaluation.feasible:
                diagnostics["rejected_evaluation"] = evaluation
                raise NumericalError("The returned selection violates original constraints.")
            if raw.objective is None:
                raise NumericalError("Solver returned a solution without an objective value.")
            reported = finite(model.offset + model.sign * raw.objective)
            if abs(reported - evaluation.objective_value) > tolerances.allowance(
                reported, evaluation.objective_value
            ):
                raise NumericalError("Solver objective disagrees with direct evaluation.")
            bound = None if raw.bound is None else finite(model.offset + model.sign * raw.bound)
            if bound is not None:
                finite(abs(bound - evaluation.objective_value))
                wrong_direction = model.sign * (bound - evaluation.objective_value)
                if wrong_direction > tolerances.allowance(bound, evaluation.objective_value):
                    raise NumericalError("Solver bound contradicts the verified objective value.")
        except (NumericalError, OverflowError) as exc:
            diagnostics["verify_seconds"] = perf_counter() - started
            diagnostics["verification_error"] = str(exc)
            diagnostics["candidate_values"] = tuple(float(v) for v in x.flat)
            return finish(Status.ERROR, termination, Verification.FAILED)
        diagnostics["verify_seconds"] = perf_counter() - started
        return finish(
            Status.OPTIMAL if raw.status == 0 else Status.FEASIBLE,
            termination,
            Verification.PASSED,
            evaluation,
            selected,
            bound,
        )
