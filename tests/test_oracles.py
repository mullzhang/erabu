"""Compare optimization with direct calculations on every small selection."""

import itertools
import math
import random
import runpy
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from erabu import Problem, Status

ROOT = Path(__file__).resolve().parents[1]
SHIPPING_DIR = ROOT / "examples" / "shipping"
SHIPPING_SCRIPT = SHIPPING_DIR / "main.py"


@pytest.mark.parametrize("relative_path", ["exercise.py", "shipping/main.py", "shifts.py"])
def test_examples(relative_path, monkeypatch):
    script = ROOT / "examples" / relative_path
    monkeypatch.setattr(sys, "argv", [str(script)])
    runpy.run_path(str(script), run_name="__main__")


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


@pytest.fixture
def shipping_csv(tmp_path):
    """A four-candidate oracle with string IDs that CSV inference could corrupt."""
    files = {
        "routes.csv": "pattern_id,ship,cost\n001,01,2\n002,01,3\n003,02,2\n004,02,3\n",
        "route_jobs.csv": "pattern_id,job\n001,NA\n001,002\n002,NA\n003,002\n003,003\n004,003\n",
        "ships.csv": "ship\n01\n02\n",
        "jobs.csv": "job\nNA\n002\n003\n",
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_shipping_cli_reads_custom_directory(shipping_csv, monkeypatch, capsys):
    monkeypatch.chdir(shipping_csv.parent)
    monkeypatch.setattr(sys, "argv", [str(SHIPPING_SCRIPT), "--data-dir", str(shipping_csv)])
    runpy.run_path(str(SHIPPING_SCRIPT), run_name="__main__")
    output = capsys.readouterr().out
    assert "Shipping (cover): OPTIMAL, cost=4.0" in output
    assert "Shipping (partition): OPTIMAL, cost=5.0" in output
    assert "Candidates: 4" in output


@pytest.mark.parametrize("exact", [True, False])
def test_shipping_all_sixteen_selections(exact, shipping_csv):
    problem = runpy.run_path(str(SHIPPING_SCRIPT))["build_problem"](shipping_csv, exact=exact)
    best = math.inf
    for x in itertools.product((0, 1), repeat=4):
        counts = (x[0] + x[1], x[0] + x[2], x[2] + x[3])
        feasible = (
            x[0] + x[1] <= 1
            and x[2] + x[3] <= 1
            and all(c == 1 if exact else c >= 1 for c in counts)
        )
        cost = sum(a * b for a, b in zip(x, (2, 3, 2, 3), strict=True))
        evaluation = problem.evaluate([f"{i + 1:03d}" for i in range(4) if x[i]])
        assert evaluation.feasible == feasible
        assert evaluation.objective_value == cost
        if feasible:
            best = min(best, cost)
    result = problem.solve()
    assert result.status is Status.OPTIMAL
    assert result.objective_value == best == (5 if exact else 4)


@pytest.mark.parametrize("exact", [True, False])
def test_shipping_csv_retains_jobs_without_routes(exact, shipping_csv):
    with (shipping_csv / "jobs.csv").open("a", encoding="utf-8") as file:
        file.write("unserved\n")
    problem = runpy.run_path(str(SHIPPING_SCRIPT))["build_problem"](shipping_csv, exact=exact)
    result = problem.solve()
    assert result.status is Status.INFEASIBLE
    assert result.bundle is None


@pytest.mark.parametrize("exact,expected", [(False, 1114), (True, 1876)])
def test_shipping_csv_solution_against_original_tables(exact, expected):
    directory = SHIPPING_DIR / "data"
    routes = pd.read_csv(directory / "routes.csv")
    membership = pd.read_csv(directory / "route_jobs.csv")
    jobs = pd.read_csv(directory / "jobs.csv")["job"].tolist()
    ships = pd.read_csv(directory / "ships.csv")["ship"].tolist()
    problem = runpy.run_path(str(SHIPPING_SCRIPT))["build_problem"](directory, exact=exact)
    assert problem.inspect().candidate_count == len(routes) == 120
    assert len(ships) == 20
    assert len(jobs) == 40
    result = problem.solve()
    assert result.status is Status.OPTIMAL
    assert result.bundle is not None
    chosen = routes[routes["pattern_id"].isin(result.bundle.pattern_ids)]
    ship_counts = Counter(chosen["ship"])
    job_counts = Counter(
        membership[membership["pattern_id"].isin(result.bundle.pattern_ids)]["job"]
    )
    assert set(ship_counts) <= set(ships)
    assert all(count <= 1 for count in ship_counts.values())
    assert set(job_counts) == set(jobs)
    assert all(count == 1 if exact else count >= 1 for count in job_counts.values())
    assert result.objective_value == chosen["cost"].sum() == expected


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
