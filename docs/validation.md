# Validation

Run `mise run verify` for formatting, lint, strict type checks, documentation
checks, the test suite, standalone examples, and isolated package verification.

The correctness tests enumerate all selections for the three-candidate exercise
and four-candidate shipping examples. Seeded six-candidate problems compare both
minimization and maximization against plain-Python calculations, including signed
pair coefficients, variance, squared targets, and a quadratic constraint.
These small-instance checks do not establish performance on larger applications.

Controlled solver replies cover optimal, feasible, infeasible, no-incumbent, and
error results. They also exercise malformed vectors, fractional selections,
incorrect auxiliary variables, failed original constraints, inconsistent
objectives and bounds, missing evidence, and solver exceptions. This avoids relying
on a machine's timing to trigger a particular solver state.

Contract tests cover missing targets, zero totals, input snapshots, label alignment,
empty selections, invalid values, and rejection before large quadratic expansion.
The README Python example is executed by `scripts/check_docs.py`; other Python
snippets are parsed, and repository-local documentation links are checked.
Public project files are checked for ASCII text to keep repository content in
English. `docs_local` is excluded from that check and from distributions.

`mise run verify:wheel` builds a source distribution and wheel, rebuilds the wheel
from the source archive, installs it with locked runtime dependencies in a fresh
environment, and runs every example using isolated Python outside the checkout.
It checks that `erabu` was imported from that environment, that `py.typed` is
included, and that local planning documents are absent from the distributions.

CI is configured to run the verification workflow on Python 3.12, 3.13, and 3.14
on Linux, and the default pinned Python on macOS. A local test run is not evidence
that the remote CI jobs have already run.

## Measuring model preparation

Run `mise run benchmark` to compare ERABU's model preparation with an indexed
handwritten Python construction. The script reports the median of three repetitions
at 1,000 and 10,000 candidates, with 100 groups. Both preserve an explicit empty
group and use the same values. The baseline constructs sparse coefficient
dictionaries; ERABU additionally validates, builds expressions, and constructs a
sparse solver matrix, so the timing includes that contract overhead.

The benchmark also solves separate 100-candidate linear and quadratic examples and
reports compilation, solving, verification, and expanded model sizes. Those timings
describe different problem classes and are not a claim that either formulation
is generally faster. No claim about real shipping or exercise workloads follows
from these synthetic examples.

## Initial local measurements

On 2026-09-30, the benchmark ran on macOS arm64 with Python 3.14.6, pandas 3.0.6,
NumPy 2.5.3, and SciPy 1.18.1. Preparation times below are medians of three runs,
with 100 groups and one additional empty target:

| Candidates | Indexed Python | ERABU preparation |
| --- | --- | --- |
| 1,000 | 0.361 ms | 3.313 ms |
| 10,000 | 1.894 ms | 29.129 ms |

The indexed baseline is faster for this preparation workload. ERABU pays for
validation, symbolic expressions, and sparse model construction; these measurements
do not support a speed advantage over handwritten indexed Python.

With a two-second solver limit, the separate 100-candidate linear model was
`OPTIMAL` at approximately 0.714286. The quadratic model introduced 4,950 auxiliary
variables and reached `FEASIBLE` with objective 16.142857 and lower bound
-2682.285714. Its optimum was not established. Binary-product linearization can
have a weak relaxation as well as a large expanded model; reducing candidates or
using a more specialized formulation may be necessary for larger quadratic work.
