"""Read CSV schedules and compare job coverage with exact partitioning."""

import argparse
from pathlib import Path

import pandas as pd

from erabu import Problem

DATA_DIR = Path(__file__).resolve().parent / "data"


def build_problem(data_dir: Path, *, exact: bool) -> Problem:
    routes = pd.read_csv(
        data_dir / "routes.csv",
        dtype={"pattern_id": str, "ship": str},
        keep_default_na=False,
    )
    membership = pd.read_csv(
        data_dir / "route_jobs.csv",
        dtype={"pattern_id": str, "job": str},
        keep_default_na=False,
    )
    ships = pd.read_csv(data_dir / "ships.csv", dtype={"ship": str}, keep_default_na=False)[
        "ship"
    ].tolist()
    jobs = pd.read_csv(data_dir / "jobs.csv", dtype={"job": str}, keep_default_na=False)[
        "job"
    ].tolist()
    problem = Problem(routes, key="pattern_id")
    problem.require(
        problem.selected.groupby("ship", universe=ships).count() <= 1,
        name="One full-period schedule per ship",
    )
    bounds = {"exactly": 1} if exact else {"at_least": 1}
    problem.cover(
        membership=membership,
        pattern="pattern_id",
        item="job",
        universe=jobs,
        name="Job coverage",
        **bounds,
    )
    problem.minimize(problem.selected.sum("cost"))
    return problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing routes.csv, route_jobs.csv, ships.csv, and jobs.csv.",
    )
    args = parser.parse_args()
    for exact in (False, True):
        problem = build_problem(args.data_dir, exact=exact)
        result = problem.solve(time_limit=30)
        mode = "partition" if exact else "cover"
        print(
            f"Shipping ({mode}): {result.status}, cost={result.objective_value}, "
            f"best_bound={result.best_bound}"
        )
        if result.bundle is None:
            raise SystemExit(f"No verified selection: {result.termination}, {result.verification}.")
        print(
            f"Candidates: {problem.inspect().candidate_count}; "
            f"selected schedules: {len(result.bundle.pattern_ids)}"
        )
        print(result.bundle.patterns.to_string(index=False))
        assert result.constraint_report is not None
        jobs = [check for check in result.constraint_report if check.name == "Job coverage"]
        print(
            f"Jobs: {len(jobs)}; covered more than once: {sum(check.value > 1 for check in jobs)}"
        )


if __name__ == "__main__":
    main()
