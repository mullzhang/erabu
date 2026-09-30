# Architecture

The public vocabulary is Problem, Pattern, Bundle, and Result. A Pattern is a
candidate row; callers do not need to construct per-row wrapper objects.

```text
Candidate and relation tables
          |
          v
Validated snapshots -> Symbolic expressions
                            |          |
                            |          +--> Direct evaluator
                            v                   ^
                     Sparse MILP model          |
                            |                   |
                            v                   |
                       SciPy / HiGHS -> Candidate selection
                                                |
                                                v
                                        Verified Result
```

| Module | Responsibility |
| --- | --- |
| `data.py` | IDs, finite coefficients, snapshots, relation validation |
| `expressions.py` | Scalar/vector arithmetic and selection aggregates |
| `_nodes.py` | Internal expression definitions shared by both execution paths |
| `compiler.py` | Conservative size inspection, polynomial expansion, sparse linearization |
| `solver.py` | SciPy/HiGHS invocation and raw result extraction |
| `evaluation.py` | Direct aggregation of selected original values |
| `problem.py` | Registration, orchestration, verification, state mapping |
| `result.py` | Public results, statuses, and numerical tolerances |

The compiler and evaluator share expression definitions but do not share compiled
coefficient evaluation. Tests also use independent plain-Python enumeration so
both paths cannot establish correctness merely by agreeing with each other.

For each required product of two binary variables, introduce a continuous variable
`y` in `[0, 1]` and retain all three constraints:

```text
y <= x[p]
y <= x[q]
y >= x[p] + x[q] - 1
```

These imply `y = x[p] * x[q]` for binary selections, regardless of coefficient
sign. Diagonal terms use `x[p] ** 2 = x[p]`. The same product is shared across
objectives and constraints. Objective constants and maximization signs are
restored when reporting values and bounds.

Quadratic expansion is delayed until compilation. The size guard runs before
pair generation and does not depend on favorable coefficient cancellation.
Large dense quadratic models need a different formulation or fewer candidates;
the library does not silently approximate or switch solvers.

The backend uses floating-point mixed-integer optimization. The direct evaluator
checks original constraints, while solver optimality still depends on its numeric
criteria. The relevant upstream contracts are
[SciPy `milp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
and [HiGHS](https://ergo-code.github.io/HiGHS/dev/#Specification).
