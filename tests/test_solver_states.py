"""Control solver replies to verify state and evidence boundaries."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from erabu import Problem, Status, Termination, Verification
from erabu.solver import SolverOutput


def model():
    p = Problem(pd.DataFrame({"id": ["p"]}), key="id")
    p.minimize(p.selected.count() + 4)
    return p


def install(monkeypatch, raw):
    monkeypatch.setattr("erabu.problem.run_solver", lambda *args, **kwargs: raw)


@pytest.mark.parametrize(
    "status,x,expected,verification",
    [
        (0, [1.0], Status.OPTIMAL, Verification.PASSED),
        (1, [1.0], Status.FEASIBLE, Verification.PASSED),
        (1, None, Status.NO_SOLUTION, Verification.NOT_RUN),
        (2, None, Status.INFEASIBLE, Verification.NOT_RUN),
        (3, None, Status.ERROR, Verification.NOT_RUN),
        (4, None, Status.ERROR, Verification.NOT_RUN),
        (0, None, Status.ERROR, Verification.NOT_RUN),
        (2, [1.0], Status.ERROR, Verification.NOT_RUN),
    ],
)
def test_state_mapping(monkeypatch, status, x, expected, verification):
    raw = SolverOutput(
        status,
        "Controlled test reply",
        None if x is None else np.array(x),
        None if x is None else 1.0,
        0 if status == 1 else 1.0,
        0.0,
    )
    install(monkeypatch, raw)
    result = model().solve(time_limit=1, node_limit=2)
    assert result.status is expected
    assert result.verification is verification
    if status == 1:
        assert result.termination is Termination.LIMIT
    if expected in (Status.OPTIMAL, Status.FEASIBLE):
        assert result.bundle.pattern_ids == ("p",)
        assert result.objective_value == 5
        assert result.absolute_gap == (1 if status == 1 else 0)
    else:
        assert result.bundle is None
        assert result.objective_value is None
        assert result.metrics is None
        assert result.constraint_report is None


@pytest.mark.parametrize(
    "change",
    [
        {"x": np.array([0.4])},
        {"x": np.array([np.nan])},
        {"x": np.array([np.inf])},
        {"x": np.array([])},
        {"x": np.array([[1.0]])},
        {"x": np.array([-1.0])},
        {"x": np.array([2.0])},
        {"objective": 7.0},
        {"objective": None},
        {"objective": np.nan},
        {"bound": 2.0},
        {"bound": np.inf},
    ],
)
def test_invalid_solution_never_becomes_a_bundle(monkeypatch, change):
    raw = SolverOutput(0, "Controlled test reply", np.array([1.0]), 1.0, 1.0, 0.0)
    install(monkeypatch, replace(raw, **change))
    result = model().solve()
    assert result.status is Status.ERROR
    assert result.verification is Verification.FAILED
    assert result.bundle is None
    assert "verification_error" in result.diagnostics


def test_near_integral_values_are_rechecked(monkeypatch):
    install(monkeypatch, SolverOutput(1, "Time limit", np.array([1.0 - 1e-8]), 1.0, 0.0, 1.0))
    p = model()
    p.require(p.selected.count() == 0, name="Must be empty")
    result = p.solve()
    assert result.status is Status.ERROR
    check = result.diagnostics["rejected_evaluation"].constraint_report[0]
    assert check.value == 1
    assert check.violation == 1


def test_wrong_auxiliary_product_is_rejected(monkeypatch):
    p = Problem(pd.DataFrame({"id": [1, 2]}), key="id")
    p.minimize(p.selected.count().square())
    install(monkeypatch, SolverOutput(0, "Wrong product", np.array([1.0, 1.0, 0.0]), 4.0, 4.0, 0.0))
    result = p.solve()
    assert result.status is Status.ERROR
    assert "Auxiliary" in result.diagnostics["verification_error"]


def test_optional_bound_stays_missing(monkeypatch):
    install(monkeypatch, SolverOutput(1, "Limit", np.array([1.0]), 1.0, None, None))
    result = model().solve()
    assert result.status is Status.FEASIBLE
    assert result.best_bound is None
    assert result.absolute_gap is None


def test_nonfinite_bound_without_solution_is_error(monkeypatch):
    install(monkeypatch, SolverOutput(1, "Limit", None, None, np.nan, None))
    assert model().solve().status is Status.ERROR


def test_solver_exception_is_distinct_from_infeasibility(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Solver unavailable")

    monkeypatch.setattr("erabu.problem.run_solver", fail)
    result = model().solve()
    assert result.status is Status.ERROR
    assert result.termination is Termination.SOLVER_ERROR
    assert result.verification is Verification.NOT_RUN
