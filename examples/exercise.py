"""Balance total exercise scores across body parts."""

import pandas as pd

from erabu import Problem, Status


def build_problem() -> Problem:
    patterns = pd.DataFrame(
        {
            "pattern_id": ["p1", "p2", "p3"],
            "slot": ["t1", "t2", "t1"],
            "arms": [2, 0, 1],
            "legs": [0, 2, 1],
        }
    )
    problem = Problem(patterns, key="pattern_id")
    selected = problem.selected
    problem.require(
        selected.groupby("slot", universe=["t1", "t2"]).count() <= 1,
        name="One exercise per slot",
    )
    totals = selected.sum(["arms", "legs"])
    problem.metric("Body totals", totals)
    problem.metric("Population variance", totals.variance())
    problem.minimize(totals.variance() - totals.mean())
    return problem


def main() -> None:
    result = build_problem().solve()
    assert result.status is Status.OPTIMAL
    assert result.bundle is not None
    assert result.bundle.pattern_ids == ("p1", "p2")
    assert result.objective_value == -2
    print(f"Exercise: {result.status}, objective={result.objective_value}")
    print(result.bundle.patterns.to_string(index=False))
    assert result.metrics is not None
    print(dict(result.metrics))


if __name__ == "__main__":
    main()
