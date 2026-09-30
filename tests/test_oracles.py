"""Compare optimization with direct calculations on every small selection."""

import itertools
import math
import random
import runpy
from pathlib import Path

import pandas as pd
import pytest

from erabu import Problem, Status

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["exercise", "shipping", "shifts"])
def test_examples(name):
    runpy.run_path(str(ROOT / "examples" / f"{name}.py"), run_name="__main__")


def test_exercise_all_eight_selections():
    problem = runpy.run_path(str(ROOT / "examples/exercise.py"))["build_problem"]()
    for x in itertools.product((0, 1), repeat=3):
        totals = (2 * x[0] + x[2], 2 * x[1] + x[2])
        mean = sum(totals) / 2
        variance = sum((v - mean) ** 2 for v in totals) / 2
        evaluation = problem.evaluate([f"p{i + 1}" for i in range(3) if x[i]])
        assert evaluation.feasible == (x[0] + x[2] <= 1)
        assert evaluation.objective_value == variance - mean
        assert evaluation.metrics["Body totals"] == dict(zip(("arms", "legs"), totals, strict=True))


@pytest.mark.parametrize("exact", [True, False])
def test_shipping_all_sixteen_selections(exact):
    problem = runpy.run_path(str(ROOT / "examples/shipping.py"))["build_problem"](exact=exact)
    best = math.inf
    for x in itertools.product((0, 1), repeat=4):
        counts = (x[0] + x[1], x[0] + x[2], x[2] + x[3])
        feasible = (
            x[0] + x[1] <= 1
            and x[2] + x[3] <= 1
            and all(c == 1 if exact else c >= 1 for c in counts)
        )
        cost = sum(a * b for a, b in zip(x, (2, 3, 2, 3), strict=True))
        evaluation = problem.evaluate([f"r{i + 1}" for i in range(4) if x[i]])
        assert evaluation.feasible == feasible
        assert evaluation.objective_value == cost
        if feasible:
            best = min(best, cost)
    result = problem.solve()
    assert result.status is Status.OPTIMAL
    assert result.objective_value == best == (5 if exact else 4)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("maximize", [False, True])
def test_random_small_models_against_independent_enumeration(seed, maximize):
    rng = random.Random(seed)
    n = 6
    a = [rng.randrange(-4, 9) / 4 for _ in range(n)]
    b = [rng.randrange(-4, 9) / 4 for _ in range(n)]
    costs = [rng.randrange(-5, 8) / 2 for _ in range(n)]
    pairs = [(p, q, rng.randrange(-4, 5) / 3) for p, q in itertools.combinations(range(n), 2)]
    patterns = pd.DataFrame({"id": range(n), "a": a, "b": b, "cost": costs})
    problem = Problem(patterns, key="id")
    s = problem.selected
    problem.require(s.count() >= 1, name="Minimum size")
    problem.require(s.count() <= 3, name="Maximum size")
    problem.require((s.sum("a") - 0.5).square() <= 4, name="Quadratic budget")
    pair_values = s.pair_sum(
        pd.DataFrame(pairs, columns=["p", "q", "v"]), left="p", right="q", value="v"
    )
    totals = s.sum(["a", "b"])
    objective = totals.variance() - totals.mean() + 0.1 * (s.sum("cost") - 2).square() + pair_values
    (problem.maximize if maximize else problem.minimize)(objective)
    possible = []
    for x in itertools.product((0, 1), repeat=n):
        ta = sum(v * k for v, k in zip(a, x, strict=True))
        tb = sum(v * k for v, k in zip(b, x, strict=True))
        cost = sum(v * k for v, k in zip(costs, x, strict=True))
        mean = (ta + tb) / 2
        direct = ((ta - mean) ** 2 + (tb - mean) ** 2) / 2 - mean
        direct += 0.1 * (cost - 2) ** 2 + sum(v * x[p] * x[q] for p, q, v in pairs)
        feasible = 1 <= sum(x) <= 3 and (ta - 0.5) ** 2 <= 4
        evaluation = problem.evaluate([i for i in range(n) if x[i]])
        assert evaluation.feasible == feasible
        assert evaluation.objective_value == pytest.approx(direct)
        if feasible:
            possible.append(direct)
    result = problem.solve()
    assert result.status is Status.OPTIMAL, result.diagnostics
    best = (max if maximize else min)(possible)
    assert result.objective_value == pytest.approx(best)
    assert result.best_bound == pytest.approx(best)
    assert result.absolute_gap == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("xp,xq", tuple(itertools.product((0, 1), repeat=2)))
@pytest.mark.parametrize("coefficient", [-3.0, 2.0])
def test_binary_products_are_exact_for_both_signs(xp, xq, coefficient):
    problem = Problem(pd.DataFrame({"id": ["p", "q"], "p": [1, 0], "q": [0, 1]}), key="id")
    problem.require(problem.selected.sum("p") == xp, name="Fix p")
    problem.require(problem.selected.sum("q") == xq, name="Fix q")
    pair = problem.selected.pair_sum(
        pd.DataFrame({"left": ["p"], "right": ["q"], "value": [coefficient]}),
        left="left",
        right="right",
        value="value",
    )
    problem.require(pair >= min(0, coefficient), name="Reuse product")
    problem.minimize(pair + 17)
    result = problem.solve()
    assert result.status is Status.OPTIMAL, result.diagnostics
    assert result.objective_value == 17 + coefficient * xp * xq
    assert result.diagnostics["pair_count"] == 1


def test_targets_and_group_alignment_and_variance_bound():
    problem = Problem(pd.DataFrame({"id": [0, 1], "group": ["a", "a"], "v": [2, 3]}), key="id")
    groups = problem.selected.groupby("group", universe=["a", "b"])
    totals = groups.sum("v")
    problem.metric("totals", totals)
    problem.metric(
        "aligned", totals + problem.selected.groupby("group", universe=["b", "a"]).sum("v")
    )
    problem.require(totals.variance() <= 2, name="Balance limit")
    problem.minimize((totals - {"b": 1, "a": 2}).square().sum())
    result = problem.solve()
    assert result.status is Status.OPTIMAL
    assert result.bundle.pattern_ids == (0,)
    assert result.objective_value == 1
    assert result.metrics["totals"] == {"a": 2, "b": 0}
    assert result.metrics["aligned"] == {"a": 4, "b": 0}


def test_usage_and_forbidden_pairs():
    p = Problem(pd.DataFrame({"id": [1, 2, 3], "benefit": [4, 3, 8]}), key="id")
    use = pd.DataFrame({"candidate": [1, 2, 3], "resource": ["r", "r", "r"], "amount": [2, 2, 5]})
    p.require(
        p.selected.related(use, pattern="candidate", item="resource", universe=["r"]).sum("amount")
        <= 4,
        name="Capacity",
    )
    p.forbid_pairs(pd.DataFrame({"a": [1], "b": [2]}), left="a", right="b", name="Conflict")
    p.maximize(p.selected.sum("benefit"))
    result = p.solve()
    assert result.status is Status.OPTIMAL
    assert result.bundle.pattern_ids == (1,)
    assert result.objective_value == 4
