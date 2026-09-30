# Contributing

All repository content is written in English, except local notes under
`docs_local/`, which are ignored by Git and excluded from package builds.

Use mise to select the pinned Python and uv versions and uv to maintain the
environment and lockfile:

```sh
mise trust
mise install
mise run setup
mise run verify
```

Useful tasks:

| Task | Purpose |
| --- | --- |
| `mise run check` | Ruff, strict mypy, language and Markdown checks |
| `mise run test` | Tests and branch coverage report |
| `mise run examples` | Exercise, shipping, and shift examples |
| `mise run build` | Wheel and source archive in `dist/` |
| `mise run verify:wheel` | Verify a rebuilt, installed wheel outside the checkout |
| `mise run benchmark` | Indexed Python comparison and solver phase timings |

Use `uv add` or `uv add --group dev` for dependency changes and commit `uv.lock`
alongside `pyproject.toml`. Do not modify the lockfile by hand. Builds use the
Hatchling backend. Python 3.12 is the minimum supported runtime; the default mise
environment is pinned independently in `mise.toml` and `.python-version`.

To check a different installed Python without replacing the default environment:

```sh
UV_PROJECT_ENVIRONMENT=.cache/py312 uv run --python 3.12 --locked pytest
```

For new mathematical operations, add a direct evaluator and a compiler path, then
compare their behavior against independent enumeration on small inputs. Preserve
the distinction between invalid input, infeasibility, a solver limit without a
solution, and a failed returned-solution check. Avoid implicit data repair,
compatibility aliases, automatic solver fallback, and unmeasured performance claims.
