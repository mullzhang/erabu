# ERABU

**Expressive Rule-Aware Bundling Utility** selects a bundle of candidate patterns
under aggregate constraints and objectives. Prepare candidates as a pandas table,
describe what their selection must achieve, and receive a verified result.

ERABU handles binary selection, coverage and partitioning, resource limits,
conflicting pairs, aggregate variance, squared targets, and pairwise preferences.
You supply feasible candidates; ERABU chooses among them. Optimality always refers
to that supplied candidate set.

## Getting started

Python 3.12 or newer is required. From a checkout, install the package into an
environment with uv:

```sh
uv venv
uv pip install .
```

For development, [mise](https://mise.jdx.dev/) manages Python and uv, while
[uv](https://docs.astral.sh/uv/) installs the versions recorded in `uv.lock`:

```sh
mise trust
mise install
mise run setup
mise run examples
mise run verify
```

## Select a balanced exercise menu

Each row is a candidate pattern. `problem.selected` describes an unknown selection;
its aggregates build expressions instead of computing values immediately.

```python
import pandas as pd

from erabu import Problem, Status

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
problem.minimize(totals.variance(ddof=0) - totals.mean())

result = problem.solve(time_limit=30)
assert result.status is Status.OPTIMAL
assert result.bundle is not None
assert result.bundle.pattern_ids == ("p1", "p2")
assert result.objective_value == -2
print(result.bundle.patterns)
```

The selected body-part totals are `(2, 2)`: mean 2, population variance 0, and
objective -2. This is variance **after summing the selected rows by column**.
It is not the sum of individual candidates' variances. All specified body parts
remain in the calculation, including those with zero scores.

Minimizing variance alone can favor selecting nothing. Add a minimum count or
total-score constraint when the application requires a minimum amount of work.
Weights are explicit; ERABU does not silently normalize objectives.

## Relations and coverage

A candidate can cover several jobs or consume several resources. Supply these
relationships in a separate table and provide the complete target `universe`.
Missing targets then remain visible instead of disappearing from the model.

| Operation | Meaning |
| --- | --- |
| `selected.sum("cost")` | Total cost of selected candidates |
| `selected.groupby("worker", universe=workers).sum("hours")` | Hours for each worker, including zero totals |
| `selected.related(usage, pattern="id", item="resource", universe=resources).sum("amount")` | Resource usage from a relation table |
| `problem.cover(..., at_least=1, name="Coverage")` | Every target occurs at least once; overlaps are allowed |
| `problem.cover(..., exactly=1, name="Partition")` | Every target occurs exactly once |
| `problem.cover(..., at_most=1, name="Packing")` | Targets occur at most once; unserved targets are allowed |
| `problem.forbid_pairs(..., name="Conflicts")` | Prohibit simultaneous selection of specified pairs |
| `selected.pair_sum(..., value="penalty")` | Sum coefficients for pairs selected together |

See [shipping](examples/shipping.py) for coverage versus partitioning and
[shifts](examples/shifts.py) for staffing, capacity, and workload balance.
The shipping example has minimum cost 4 with overlapping coverage and 5 with
exact partitioning. It also limits each ship to one full-period schedule.

## Results you can inspect

`result.bundle` contains only a verified selection. It is `None` when no verified
solution is available; a valid empty selection is a Bundle with no rows.
`bundle.pattern_ids` preserves input row order, and `bundle.patterns` returns a copy
of the corresponding original rows.

| Status | Meaning |
| --- | --- |
| `OPTIMAL` | The solver reported optimality within its numerical criteria, and verification passed |
| `FEASIBLE` | A verified incumbent is available, without established optimality |
| `INFEASIBLE` | The supplied model was established to have no feasible selection |
| `NO_SOLUTION` | A solver limit was reached without an incumbent |
| `ERROR` | Solver failure or a returned solution that failed verification |

Inspect `termination` and `verification` separately. A time limit can return a
useful `FEASIBLE` result. `NO_SOLUTION` does not establish infeasibility.
Missing objectives, metrics, and bounds remain `None` instead of becoming zero.

Named metrics are available in `result.metrics`. Each entry in
`result.constraint_report` identifies the condition, target, actual value, bounds,
violation, tolerance, and pass/fail outcome. `best_bound` is a lower bound for
minimization and an upper bound for maximization. `absolute_gap` uses the original
objective scale, including constants.

`problem.evaluate(pattern_ids)` checks an existing selection without solving and
does not make any optimality claim.

## Solver and scale

ERABU uses [SciPy's `milp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
interface to HiGHS. Binary products in quadratic objectives and constraints are
linearized exactly, with shared continuous auxiliary variables. Hard constraints
are retained as constraints rather than replaced by penalty weights.

Dense quadratic expressions can require one auxiliary per pair of candidates.
At 1,000 candidates, that is up to 499,500 auxiliaries and 1,498,500 extra rows.
`problem.inspect()` provides a conservative estimate before expansion;
`solve(max_pair_terms=100_000)` rejects larger estimates before generating pairs.
The limit can be changed explicitly and does not guarantee a runtime or memory
budget. Display-only metrics do not add solver variables.

`time_limit` and `node_limit` limit the solver, not total API execution time.
Diagnostics separate candidate snapshot validation, model compilation, solving,
and verification. See [the API reference](docs/api.md) for numerical tolerances.

## Scope

The current release accepts a fixed candidate set and selects each candidate at
most once. It supports affine expressions, affine squares, fixed-universe
population variance, and pair sums. It does not generate candidates, implement
column generation, optimize arbitrary Python functions, or provide a QUBO backend.

Every candidate's internal feasibility remains the caller's responsibility.
Examples use small synthetic data; they are not reproductions of external shipping
or exercise studies. No large-instance performance claim is implied.

## Documentation

- [API reference](docs/api.md)
- [Architecture](docs/architecture.md)
- [Validation and benchmarks](docs/validation.md)
- [Contributing](CONTRIBUTING.md)
