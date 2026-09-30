"""Choose full-period work schedules and balance hours across a fixed workforce."""

import pandas as pd

from erabu import Problem, Status


def build_problem() -> Problem:
    patterns = pd.DataFrame(
        {
            "pattern_id": ["alex_am", "alex_both", "blair_pm", "blair_off"],
            "worker": ["alex", "alex", "blair", "blair"],
            "hours": [4, 8, 4, 0],
            "cost": [5, 6, 5, 0],
        }
    )
    coverage = pd.DataFrame(
        {
            "pattern_id": ["alex_am", "alex_both", "alex_both", "blair_pm"],
            "slot": ["am", "am", "pm", "pm"],
            "equipment": [1, 1, 1, 1],
        }
    )
    problem = Problem(patterns, key="pattern_id")
    workers = problem.selected.groupby("worker", universe=["alex", "blair"])
    problem.require(workers.count() == 1, name="One schedule per worker")
    problem.cover(
        membership=coverage,
        pattern="pattern_id",
        item="slot",
        universe=["am", "pm"],
        exactly=1,
        name="Staffing requirements",
    )
    equipment = problem.selected.related(
        coverage, pattern="pattern_id", item="slot", universe=["am", "pm"]
    ).sum("equipment")
    problem.require(equipment <= {"am": 1, "pm": 1}, name="Equipment capacity")
    hours = workers.sum("hours")
    problem.metric("Hours per worker", hours)
    problem.metric("Cost", problem.selected.sum("cost"))
    problem.minimize(hours.variance() + problem.selected.sum("cost"))
    return problem


def main() -> None:
    result = build_problem().solve()
    assert result.status is Status.OPTIMAL
    assert result.bundle is not None
    assert result.bundle.pattern_ids == ("alex_am", "blair_pm")
    assert result.objective_value == 10
    print(f"Shifts: {result.status}, objective={result.objective_value}")
    print(result.bundle.patterns.to_string(index=False))


if __name__ == "__main__":
    main()
