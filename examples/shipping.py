"""Compare coverage and exact partitioning of jobs by full-period ship schedules."""

import pandas as pd

from erabu import Problem, Status


def build_problem(*, exact: bool) -> Problem:
    routes = pd.DataFrame(
        {
            "pattern_id": ["r1", "r2", "r3", "r4"],
            "ship": ["s1", "s1", "s2", "s2"],
            "cost": [2, 3, 2, 3],
        }
    )
    jobs = pd.DataFrame(
        {
            "pattern_id": ["r1", "r1", "r2", "r3", "r3", "r4"],
            "job": ["a", "b", "a", "b", "c", "c"],
        }
    )
    problem = Problem(routes, key="pattern_id")
    problem.require(
        problem.selected.groupby("ship", universe=["s1", "s2"]).count() <= 1,
        name="One full-period schedule per ship",
    )
    bounds = {"exactly": 1} if exact else {"at_least": 1}
    problem.cover(
        membership=jobs,
        pattern="pattern_id",
        item="job",
        universe=["a", "b", "c"],
        name="Job coverage",
        **bounds,
    )
    problem.minimize(problem.selected.sum("cost"))
    return problem


def main() -> None:
    for exact, expected in ((False, 4), (True, 5)):
        result = build_problem(exact=exact).solve()
        assert result.status is Status.OPTIMAL
        assert result.bundle is not None
        assert result.objective_value == expected
        print(f"Shipping ({'partition' if exact else 'cover'}): cost={result.objective_value}")
        print(result.bundle.patterns.to_string(index=False))
        assert result.constraint_report is not None
        for check in result.constraint_report:
            if check.name == "Job coverage":
                print(f"  Job {check.target}: {check.value:g} selected schedules")


if __name__ == "__main__":
    main()
