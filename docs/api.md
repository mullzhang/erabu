# API reference

## Problem and input snapshots

`Problem(patterns: pandas.DataFrame, *, key: str)` snapshots the supplied table.
Each row is one candidate. Column names must be unique strings. Candidate IDs must
be unique, nonmissing strings or integers, with no mixed types or automatic casts.
The DataFrame index is not used as the ID. NumPy integer IDs are accepted.

Keep nested mutable payloads outside the table and join them back by selected ID.
Changing an input table after it has been registered does not change the model.
Changing a returned `bundle.patterns` copy does not change the saved result.

Referenced numeric columns must contain finite real numbers. Missing values,
numeric strings, booleans, and infinities are rejected. Negative costs, scores,
and pair coefficients are allowed. Relation quantities must be nonnegative.
Unused scalar metadata columns are preserved in the output.

An empty candidate table is valid when it has the required columns. Its only
selection is the empty bundle, which is evaluated without calling the solver.

## Aggregates

| Expression | Result |
| --- | --- |
| `selected.count()` | Scalar candidate count |
| `selected.sum("cost")` | Scalar sum |
| `selected.sum(["a", "b"])` | Vector indexed by the explicit column names |
| `selected.groupby("group", universe=groups).count()` | Vector of selected candidate counts |
| `selected.groupby("group", universe=groups).sum("value")` | Vector of selected column totals |
| `selected.related(table, pattern="id", item="target", universe=targets).count()` | Vector of selected relation counts |
| `selected.related(table, pattern="id", item="target", universe=targets).sum("amount")` | Vector of selected quantities |

`universe` is an ordered iterable of unique, nonmissing string or integer IDs of a
single type. It is mandatory for grouping and relation aggregation. Targets without
any candidates remain present with value zero. An input group or relation target
outside this universe is an error. Strings and unordered sets are not universes.

Relation rows must reference known candidates. Repeated `(candidate, target)` rows
are rejected; aggregate repeated quantities into one row before passing the table.
Do not infer the required universe solely from the relation table if targets may
have no candidates.

Vectors support `sum()`, `mean()`, `variance(ddof=0)`, and elementwise `square()`.
The mean and variance are across their **fixed labels**, not the selected rows.
Empty vector sums are zero; empty means and variances are undefined and rejected.
Column lists for `selected.sum(...)` must be nonempty.

Arithmetic supports addition, subtraction, negation, and multiplication or division
by numeric constants. Scalars broadcast across vectors. Constant vectors are
mappings from labels to numbers, with an exact key match. Two vector expressions
must have the same label set and are aligned by label, not position.

`square()` and `variance()` accept only affine inputs. Multiplying two symbolic
expressions, higher-degree expressions, and selected-row statistics with a variable
denominator are unsupported. Zero division and non-finite arithmetic are errors.

## Conditions and objectives

`require(condition, *, name)` registers a comparison made with `<=`, `>=`, or `==`.
Vector comparisons produce one condition per target. Both sides may be expressions
from the same problem, including quadratic expressions. Comparisons with literal
scalars and labeled constant vectors are also supported.

Use separate registrations for upper and lower limits:

```python
problem.require(selected.count() >= 2, name="Minimum size")
problem.require(selected.count() <= 5, name="Maximum size")
```

Symbolic expressions and comparisons cannot be converted to booleans. Python's
`and`, `or`, and chained comparisons are therefore not condition combinators.
Strict inequalities are unsupported. Names must be nonempty and unique within
the conditions or metrics respectively. Registrations do not silently overwrite.

`minimize(expression)` or `maximize(expression)` registers exactly one scalar
objective. An objective is required before `solve()`, `inspect()`, or `evaluate()`.
For a feasibility-only search, explicitly minimize `selected.count() * 0`.
Weighted sums are ordinary expression arithmetic. The same expression can be
used in both an objective and a constraint.

`metric(name, expression)` registers a scalar or vector for reporting. Metrics
are evaluated directly after selection and do not expand the solver model.

## Coverage and pairs

`cover(*, membership, pattern, item, universe, name, exactly=None,
at_least=None, at_most=None)` constrains the count of selected candidate-target
relations. Supply at least one nonnegative integer bound. `exactly` is exclusive;
`at_least` and `at_most` may be combined. A missing mandatory target makes a valid
problem infeasible; it is not dropped as an absent row.

`forbid_pairs(table, *, left, right, name)` prohibits selecting both candidates in
each pair. `selected.pair_sum(table, *, left, right, value)` returns a quadratic
expression that includes the coefficient only when both candidates are selected.
Positive and negative coefficients are allowed.

Pairs are undirected. Unknown IDs, self-pairs, duplicates, and reversed duplicates
within one table are errors. Different expressions can use the same pair; their
coefficients are composed and the compiled model shares its auxiliary variable.
Coverage, resource capacity, group limits, and pair conflicts are independent rules.

## Solving and inspection

`solve(*, time_limit=None, node_limit=None, max_pair_terms=100_000,
tolerances=Tolerances())` returns a `Result`.

- `time_limit` is a nonnegative number of solver seconds, or `None` for no limit.
- `node_limit` is a nonnegative integer, or `None` for no node limit.
- `max_pair_terms` is a nonnegative integer bounding a conservative estimate of
  pair expansion. The estimate may exceed the eventual number of distinct nonzero
  pairs, because separate terms can cancel or share pairs. Increase the limit only
  after inspecting the model's scale.
- The solver's MIP relative gap target is explicitly zero. Its other numerical
  criteria still apply; this is not an exact rational-arithmetic certificate.

`inspect()` returns a `ModelInfo` with `candidate_count`, `constraint_count`,
`pair_terms_upper_bound`, and `has_quadratic`. It does not expand pair terms or call
the solver. A quadratic expression can reduce to a linear model on binary variables.

Invalid inputs and unsupported expressions raise `TypeError` or `ValueError`.
`ModelSizeError` is a `ValueError` for the expansion limit. `NumericalError` is a
`ValueError` for non-finite model arithmetic. These pre-solve errors are not solver
infeasibility reports. Runtime solver errors and failed returned-solution checks
produce `Result(status=Status.ERROR)` with diagnostic evidence.

## Numerical verification

`Tolerances(absolute=1e-6, relative=1e-9, integrality=1e-6)` controls independent
verification. Values must be finite and nonnegative; integrality must be below 0.5.

For real-valued comparisons and objective agreement, the allowance is
`absolute + relative * max(abs(left), abs(right))`. Each condition report records
the actual allowance. Integer-valued comparisons with constant bounds are checked
without floating-point slack. Compilation also tightens integer-valued variable
parts to integer bounds. Rescale data if coefficient magnitudes make the default
tolerances inappropriate; automatic objective rescaling is not performed.

Returned selections must be within the integrality tolerance of zero or one.
Auxiliary variables must agree with the rounded binary products. ERABU then
recomputes every original condition and the objective. A failed check publishes no
Bundle and stores the reason in diagnostics. Neither rounding an arbitrary
fractional vector nor returning a penalized constraint violation counts as success.

## Results and direct evaluation

`Result` exposes these fields:

| Field | Meaning |
| --- | --- |
| `status` | `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `NO_SOLUTION`, or `ERROR` |
| `termination` | `COMPLETED`, `LIMIT`, or `SOLVER_ERROR` |
| `verification` | `PASSED`, `FAILED`, or `NOT_RUN` |
| `bundle` | A verified `Bundle`, or `None` |
| `objective_value` | Direct evaluation in the original objective scale, or `None` |
| `metrics` | Read-only named values; vectors are read-only labeled mappings |
| `constraint_report` | A tuple of `ConstraintCheck` records, or `None` |
| `best_bound` | Lower bound for minimization, upper bound for maximization, or `None` |
| `absolute_gap` | Absolute difference between objective and bound, or `None` |
| `diagnostics` | Raw solver status/message, sizes, tolerances, timings, and rejected-solution evidence |

`ConstraintCheck` contains `name`, `target`, `value`, `lower`, `upper`, `violation`,
`tolerance`, and `satisfied`. A coverage range produces separate lower and upper
checks under the same name. Unbounded sides are `None`.

Solver limit details are retained in its original message when the SciPy status
does not distinguish time and node limits. A missing incumbent does not imply
infeasibility. `OPTIMAL` is restricted to the supplied candidates and subject to
solver numerical criteria. Multiple optimal bundles may exist; the particular one
returned is not guaranteed, but its row ordering follows the input table.

`evaluate(pattern_ids, *, tolerances=Tolerances())` returns an `Evaluation` with
`feasible`, `objective_value`, `metrics`, and `constraint_report`, without solving.
Unlike a successful solve result, this report can describe an infeasible selection.
Duplicate or unknown IDs are errors. It makes no optimality claim.
