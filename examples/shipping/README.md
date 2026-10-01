# Shipping from CSV

This directory contains the runnable example in [main.py](main.py), its CSV input
data, and this guide. Run all domain examples with `mise run examples`.

The bundled CSV files in `data/` contain 120 candidate
full-period schedules for 20 ships and 40 jobs, with 380 schedule-job relations.
Each ship has six alternatives. Its first alternative covers a dedicated pair of
jobs, so those 20 schedules provide a feasible exact partition. The remaining
alternatives cover between two and five jobs and connect jobs across those pairs.
The tables were generated with the fixed random seed `20261001`.

```sh
uv run --locked python examples/shipping/main.py
```

The default directory is resolved relative to `main.py`, so the script can
also be launched from another working directory. To supply a different problem,
put the following four files in one directory and pass `--data-dir`:

```sh
uv run --locked python examples/shipping/main.py --data-dir /path/to/shipping-data
```

| File | Required columns | Meaning |
| --- | --- | --- |
| [routes.csv](data/routes.csv) | `pattern_id,ship,cost` | One row per candidate schedule, with a unique ID, ship, and finite numeric cost |
| [route_jobs.csv](data/route_jobs.csv) | `pattern_id,job` | One row for each job covered by a schedule; duplicate pairs are invalid |
| [ships.csv](data/ships.csv) | `ship` | Complete list of unique ship IDs, including ships without candidates |
| [jobs.csv](data/jobs.csv) | `job` | Complete list of unique required job IDs, including jobs without candidates |

Files use comma delimiters and header rows. All IDs are read as strings, preserving
leading zeros and IDs such as `NA`. Every schedule must reference a listed ship;
every relation must reference a listed schedule and job. ERABU validates those
relationships and costs when constructing the problem. A required job without a
candidate stays in the model and makes the problem infeasible.

For example, these candidate and relation rows describe one schedule covering
two jobs:

```csv
pattern_id,ship,cost
s01_r1,s01,80
```

```csv
pattern_id,job
s01_r1,j01
s01_r1,j02
```

The script solves two models. Both allow at most one schedule per ship and minimize
total cost. Coverage requires each job at least once; partitioning requires each
job exactly once. Each solve has a 30-second solver limit. The output includes the
status, objective, best bound, selected rows, and the number of jobs covered more
than once. A `FEASIBLE` result has a verified selection but does not establish
optimality; a result without a verified selection exits with a nonzero status.

The bundled coverage optimum is 1,114 and the partitioning optimum is 1,876.
Custom inputs can produce different objectives and termination states. These
synthetic tables illustrate selection among supplied schedules; the caller remains
responsible for each schedule's internal feasibility.
