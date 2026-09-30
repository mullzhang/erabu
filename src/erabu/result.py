"""Public, solver-independent result contracts."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from .data import Label, number

type MetricValue = float | Mapping[Label, float]


class Status(StrEnum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    NO_SOLUTION = "NO_SOLUTION"
    ERROR = "ERROR"


class Termination(StrEnum):
    COMPLETED = "COMPLETED"
    LIMIT = "LIMIT"
    SOLVER_ERROR = "SOLVER_ERROR"


class Verification(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True)
class Tolerances:
    """Post-solve tolerances, independent of the solver's internal tolerances."""

    absolute: float = 1e-6
    relative: float = 1e-9
    integrality: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("absolute", "relative", "integrality"):
            value = number(getattr(self, name), context=f"{name} tolerance")
            if value < 0:
                raise ValueError("Tolerances must be nonnegative.")
            object.__setattr__(self, name, value)
        if self.integrality >= 0.5:
            raise ValueError("Integrality tolerance must be less than 0.5.")

    def allowance(self, left: float, right: float) -> float:
        return self.absolute + self.relative * max(abs(left), abs(right))


@dataclass(frozen=True)
class ModelInfo:
    candidate_count: int
    constraint_count: int
    pair_terms_upper_bound: int
    has_quadratic: bool


@dataclass(frozen=True)
class ConstraintCheck:
    name: str
    target: Label | tuple[Label, Label] | None
    value: float
    lower: float | None
    upper: float | None
    violation: float
    tolerance: float
    satisfied: bool


@dataclass(frozen=True)
class Evaluation:
    feasible: bool
    objective_value: float
    metrics: Mapping[str, MetricValue]
    constraint_report: tuple[ConstraintCheck, ...]


@dataclass(frozen=True)
class Bundle:
    pattern_ids: tuple[Label, ...]
    _patterns: pd.DataFrame = field(repr=False, compare=False)

    @property
    def patterns(self) -> pd.DataFrame:
        """Return a copy so edits cannot change the saved selection."""
        return self._patterns.copy(deep=True)


@dataclass(frozen=True)
class Result:
    status: Status
    termination: Termination
    verification: Verification
    bundle: Bundle | None
    objective_value: float | None
    metrics: Mapping[str, MetricValue] | None
    constraint_report: tuple[ConstraintCheck, ...] | None
    best_bound: float | None
    absolute_gap: float | None
    diagnostics: Mapping[str, object] = field(repr=False)
