"""Measure preparation overhead against indexed Python, with explicit workloads."""

import json
from statistics import median
from time import perf_counter

import pandas as pd

from erabu import Problem, Status
from erabu.compiler import compile_model


def handwritten(table, universe):
    groups = {group: {} for group in universe}
    objective = []
    for index, (group, cost) in enumerate(
        table[["group", "cost"]].itertuples(index=False, name=None)
    ):
        groups[group][index] = 1.0
        objective.append(cost)
    return objective, list(groups.values())


def erabu_model(table, universe):
    problem = Problem(table, key="id")
    problem.require(
        problem.selected.groupby("group", universe=universe).count() <= 1, name="Group limits"
    )
    problem.minimize(problem.selected.sum("cost"))
    return compile_model(
        len(table), problem._objective(), problem._rules, maximize=False, max_pair_terms=0
    )


def measure(function, *args):
    times = []
    for _ in range(3):
        start = perf_counter()
        result = function(*args)
        times.append(perf_counter() - start)
    return median(times), result


def main():
    preparation = []
    for size in (1_000, 10_000):
        table = pd.DataFrame(
            {
                "id": range(size),
                "group": [i % 100 for i in range(size)],
                "cost": [(i % 13 - 6) / 7 for i in range(size)],
            }
        )
        universe = list(range(101))
        plain_seconds, (costs, rows) = measure(handwritten, table, universe)
        erabu_seconds, compiled = measure(erabu_model, table, universe)
        assert list(compiled.objective) == costs
        matrix = compiled.matrix.tocsr()
        for row, expected in enumerate(rows):
            start, end = matrix.indptr[row : row + 2]
            assert (
                dict(zip(matrix.indices[start:end], matrix.data[start:end], strict=True))
                == expected
            )
        preparation.append(
            {
                "candidates": size,
                "groups": 100,
                "empty_targets": 1,
                "repetitions": 3,
                "indexed_python_median_seconds": plain_seconds,
                "erabu_median_seconds": erabu_seconds,
            }
        )
    solves = []
    for quadratic in (False, True):
        table = pd.DataFrame(
            {
                "id": range(100),
                "group": [i % 20 for i in range(100)],
                "cost": [(i % 13 + 1) / 7 for i in range(100)],
                "a": [i % 7 for i in range(100)],
                "b": [(i + 3) % 7 for i in range(100)],
            }
        )
        problem = Problem(table, key="id")
        problem.require(
            problem.selected.groupby("group", universe=range(20)).count() <= 1, name="Group limits"
        )
        problem.require(problem.selected.count() >= 5, name="Minimum size")
        objective = problem.selected.sum("cost")
        if quadratic:
            objective += problem.selected.sum(["a", "b"]).variance()
        problem.minimize(objective)
        result = problem.solve(time_limit=2)
        assert result.status is not Status.ERROR, result.diagnostics
        solves.append(
            {
                "quadratic": quadratic,
                "candidates": 100,
                "time_limit_seconds": 2,
                "status": result.status,
                "objective": result.objective_value,
                "best_bound": result.best_bound,
                **{
                    name: result.diagnostics[name]
                    for name in (
                        "compile_seconds",
                        "solve_seconds",
                        "verify_seconds",
                        "variable_count",
                        "constraint_count",
                        "pair_count",
                    )
                },
            }
        )
    print(json.dumps({"preparation": preparation, "solves": solves}, indent=2))


if __name__ == "__main__":
    main()
