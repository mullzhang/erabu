"""Input contracts and expression semantics, including empty domains."""

import numpy as np
import pandas as pd
import pytest

from erabu import ModelSizeError, NumericalError, Problem, Status, Tolerances


def basic():
    return Problem(pd.DataFrame({"id": ["p", "q"], "value": [2, 3], "group": ["a", "b"]}), key="id")


@pytest.mark.parametrize("ids", [["p", "p"], [None], [float("nan")], [True], [1.0], [1, "p"]])
def test_invalid_candidate_ids(ids):
    with pytest.raises(ValueError):
        Problem(pd.DataFrame({"id": pd.Series(ids, dtype=object)}), key="id")


@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf, "2", True, pd.NA])
def test_invalid_numeric_values(value):
    p = Problem(pd.DataFrame({"id": [1], "v": pd.Series([value], dtype=object)}), key="id")
    with pytest.raises((TypeError, ValueError)):
        p.selected.sum("v")


def test_invalid_tables():
    with pytest.raises(TypeError):
        Problem([{"id": 1}], key="id")
    with pytest.raises(ValueError, match="Column names"):
        Problem(pd.DataFrame([[1, 2]], columns=["id", "id"]), key="id")
    with pytest.raises(ValueError, match="nested"):
        Problem(pd.DataFrame({"id": [1], "payload": [[1, 2]]}), key="id")
    with pytest.raises(ValueError, match="Unknown column"):
        Problem(pd.DataFrame({"other": [1]}), key="id")


@pytest.mark.parametrize("universe", [["a", "a"], ["a", None], ["a", 2], ["a"], "ab", {"a", "b"}])
def test_invalid_group_universe(universe):
    with pytest.raises((TypeError, ValueError)):
        basic().selected.groupby("group", universe=universe)


def test_missing_group_value_or_column():
    with pytest.raises(ValueError):
        basic().selected.groupby("absent", universe=[])
    p = Problem(pd.DataFrame({"id": [1], "group": [None]}), key="id")
    with pytest.raises(ValueError):
        p.selected.groupby("group", universe=["a"])


@pytest.mark.parametrize(
    "rows", [[("p", "a"), ("p", "a")], [("missing", "a")], [("p", "missing")], [("p", None)]]
)
def test_invalid_relations(rows):
    with pytest.raises(ValueError):
        basic().selected.related(
            pd.DataFrame(rows, columns=["p", "r"]), pattern="p", item="r", universe=["a"]
        )


def test_quantities_must_be_nonnegative():
    relation = basic().selected.related(
        pd.DataFrame({"p": ["p"], "r": ["a"], "v": [-1]}), pattern="p", item="r", universe=["a"]
    )
    with pytest.raises(ValueError, match="nonnegative"):
        relation.sum("v")


@pytest.mark.parametrize("pairs", [[("p", "p")], [("p", "q"), ("q", "p")], [("p", "missing")]])
def test_invalid_pairs(pairs):
    with pytest.raises(ValueError):
        basic().forbid_pairs(
            pd.DataFrame(pairs, columns=["p", "q"]), left="p", right="q", name="Conflict"
        )


@pytest.mark.parametrize(
    "bounds",
    [
        {},
        {"exactly": 1, "at_most": 1},
        {"at_least": 2, "at_most": 1},
        {"exactly": -1},
        {"at_least": 1.5},
        {"at_most": True},
    ],
)
def test_invalid_coverage_bounds(bounds):
    with pytest.raises((TypeError, ValueError)):
        basic().cover(
            membership=pd.DataFrame({"p": ["p"], "r": ["a"]}),
            pattern="p",
            item="r",
            universe=["a"],
            name="Coverage",
            **bounds,
        )


def test_missing_targets_are_constraints_not_input_errors():
    p = basic()
    p.cover(
        membership=pd.DataFrame({"p": ["p"], "r": ["a"]}),
        pattern="p",
        item="r",
        universe=["a", "unreachable"],
        exactly=1,
        name="Coverage",
    )
    p.minimize(p.selected.count())
    result = p.solve()
    assert result.status is Status.INFEASIBLE
    assert result.bundle is None
    assert result.objective_value is None
    report = p.evaluate(["p"]).constraint_report
    assert report[1].target == "unreachable"
    assert report[1].value == 0
    assert report[1].violation == 1


@pytest.mark.parametrize("feasible", [True, False])
def test_empty_candidates(feasible):
    p = Problem(pd.DataFrame({"id": [], "a": [], "b": []}), key="id")
    p.require(p.selected.count() >= (0 if feasible else 1), name="Minimum")
    p.metric("Totals", p.selected.sum(["a", "b"]))
    p.minimize(p.selected.sum(["a", "b"]).variance() + 7)
    result = p.solve()
    if feasible:
        assert result.status is Status.OPTIMAL
        assert result.bundle.pattern_ids == ()
        assert result.bundle.patterns.empty
        assert result.metrics["Totals"] == {"a": 0, "b": 0}
        assert result.objective_value == result.best_bound == 7
    else:
        assert result.status is Status.INFEASIBLE
        assert result.bundle is None


def test_valid_empty_bundle_is_not_no_solution():
    p = basic()
    p.minimize(p.selected.count())
    result = p.solve()
    assert result.status is Status.OPTIMAL
    assert result.bundle is not None
    assert result.bundle.pattern_ids == ()


def test_empty_universe_mean_and_variance_are_undefined():
    p = Problem(pd.DataFrame({"id": [], "g": []}), key="id")
    counts = p.selected.groupby("g", universe=[]).count()
    with pytest.raises(ValueError):
        counts.mean()
    with pytest.raises(ValueError):
        counts.variance()
    p.require(counts == 0, name="Vacuous")
    p.minimize(counts.sum())
    assert p.solve().status is Status.OPTIMAL


def test_snapshot_and_output_copy():
    table = pd.DataFrame({"id": ["p", "q"], "value": [2, 3]})
    membership = pd.DataFrame({"p": ["p"], "r": ["a"], "v": [2]})
    p = Problem(table, key="id")
    totals = p.selected.related(membership, pattern="p", item="r", universe=["a"]).sum("v")
    p.require(totals == 2, name="Required use")
    p.minimize(p.selected.sum("value"))
    table.loc[0, "value"] = 100
    membership.loc[0, "v"] = 999
    result = p.solve()
    assert result.objective_value == 2
    exposed = result.bundle.patterns
    exposed.loc[0, "value"] = 999
    assert result.bundle.patterns.loc[0, "value"] == 2


def test_registration_and_symbolic_truth():
    p = basic()
    count = p.selected.count()
    with pytest.raises(ValueError, match="objective"):
        p.solve()
    for value in (count, count <= 1, p.selected.sum(["value"])):
        with pytest.raises(TypeError, match="truth"):
            bool(value)
    with pytest.raises(TypeError):
        p.require(True, name="Boolean")
    with pytest.raises(TypeError):
        p.require(0 <= count <= 1, name="Chained")
    p.require(count <= 1, name="Limit")
    with pytest.raises(ValueError, match="Duplicate"):
        p.require(count <= 2, name="Limit")
    with pytest.raises(ValueError):
        p.require(count == 1, name=" ")
    p.metric("Size", count)
    with pytest.raises(ValueError):
        p.metric("Size", count)
    with pytest.raises(TypeError):
        p.minimize(p.selected.sum(["value"]))
    with pytest.raises(TypeError):
        p.metric("Literal", 2)
    p.minimize(count)
    with pytest.raises(ValueError):
        p.maximize(count)


def test_expression_errors_and_label_alignment():
    p, other = basic(), basic()
    a = p.selected.sum("value")
    v = p.selected.sum(["value"])
    for operation in (
        lambda: a + other.selected.count(),
        lambda: v + other.selected.sum(["value"]),
        lambda: p.require(other.selected.count() == 1, name="Foreign"),
        lambda: p.metric("Foreign", other.selected.count()),
    ):
        with pytest.raises(ValueError, match="different problems"):
            operation()
    for operation in (
        lambda: v + {"unknown": 2},
        lambda: v + p.selected.sum(["id"]),
        lambda: p.selected.sum([]),
        lambda: p.selected.sum(["value", "value"]),
        lambda: v.variance(ddof=1),
        lambda: a.square().square(),
        lambda: v.square().variance(),
    ):
        with pytest.raises((ValueError, TypeError)):
            operation()
    with pytest.raises(TypeError):
        a * a
    with pytest.raises(ZeroDivisionError):
        a / 0
    with pytest.raises(ValueError):
        a + np.inf
    p.minimize(a)
    for ids in (["p", "p"], ["unknown"], [True]):
        with pytest.raises(ValueError):
            p.evaluate(ids)


def test_vector_broadcasting_and_reverse_arithmetic():
    p = basic()
    v = p.selected.sum(["value"])
    a = p.selected.count()
    p.metric("Arithmetic", ((2 + v) - 1) * 2 / 2)
    p.metric("Reversed", 10 - v)
    p.metric("Scalar first", a + v)
    p.metric("Scalar difference", a - v)
    p.require(1 <= v, name="Reverse comparison")
    p.require(a <= v, name="Scalar vector comparison")
    p.minimize(4 - a)
    e = p.evaluate(["p"])
    assert e.feasible
    assert e.objective_value == 3
    assert e.metrics == {
        "Arithmetic": {"value": 3},
        "Reversed": {"value": 8},
        "Scalar first": {"value": 3},
        "Scalar difference": {"value": -1},
    }


def test_model_size_guard_precedes_expansion(monkeypatch):
    import erabu.compiler as compiler

    p = Problem(pd.DataFrame({"id": range(10_000), "a": [1] * 10_000, "b": [0] * 10_000}), key="id")
    p.minimize(p.selected.sum(["a", "b"]).variance())
    assert p.inspect().pair_terms_upper_bound == 49_995_000

    def unexpected(*args):
        pytest.fail("Quadratic expansion must not start")

    monkeypatch.setattr(compiler, "squared", unexpected)
    with pytest.raises(ModelSizeError, match="49,995,000"):
        p.solve()


def test_metrics_do_not_expand_and_limits_are_configurable():
    p = basic()
    p.metric("Variance", p.selected.sum(["value"]).variance())
    p.minimize(p.selected.count())
    assert p.inspect().pair_terms_upper_bound == 0
    assert p.solve(max_pair_terms=0).status is Status.OPTIMAL
    q = basic()
    q.minimize(q.selected.count().square())
    with pytest.raises(ModelSizeError):
        q.solve(max_pair_terms=0)
    assert q.solve(max_pair_terms=1).status is Status.OPTIMAL


@pytest.mark.parametrize(
    "kwargs",
    [{"max_pair_terms": -1}, {"node_limit": True}, {"time_limit": -1}, {"time_limit": np.nan}],
)
def test_invalid_solver_options(kwargs):
    p = basic()
    p.minimize(p.selected.count())
    with pytest.raises((ValueError, TypeError)):
        p.solve(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"absolute": -1}, {"relative": np.inf}, {"integrality": 0.5}, {"integrality": True}]
)
def test_invalid_tolerances(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Tolerances(**kwargs)


def test_integer_counts_are_checked_without_float_tolerance():
    p = basic()
    p.require(p.selected.count() <= 0.9999999999, name="Strict size")
    p.minimize(p.selected.count())
    check = p.evaluate(["p"]).constraint_report[0]
    assert check.tolerance == 0
    assert not check.satisfied
    result = p.solve()
    assert result.status is Status.OPTIMAL
    assert result.bundle.pattern_ids == ()


@pytest.mark.parametrize("bound", [0.9999999999, 1.5, 1.0000000001])
def test_fractional_integer_equalities_are_infeasible(bound):
    p = basic()
    p.require(p.selected.count() == bound, name="Exact count")
    p.minimize(p.selected.count())
    assert p.solve().status is Status.INFEASIBLE


def test_numerical_overflow_is_not_silently_accepted():
    p = Problem(pd.DataFrame({"id": [1], "value": [1e308]}), key="id")
    p.minimize(p.selected.sum("value").square())
    with pytest.raises(NumericalError):
        p.solve()
    with pytest.raises(NumericalError):
        p.evaluate([1])


def test_combined_coverage_bounds_and_integer_ids():
    p = Problem(pd.DataFrame({"id": [2, 1], "score": [5, 4]}), key="id")
    p.cover(
        membership=pd.DataFrame({"p": [1, 2], "r": [1, 1]}),
        pattern="p",
        item="r",
        universe=[1],
        at_least=1,
        at_most=1,
        name="Range",
    )
    p.maximize(p.selected.sum("score"))
    result = p.solve()
    assert result.status is Status.OPTIMAL
    assert result.bundle.pattern_ids == (2,)
