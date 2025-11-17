import numpy as np
import pytest

from ssms.support_utils import utils


@pytest.mark.parametrize(
    "bounds, expected",
    [
        ((0.0, 1.0), set()),
        (("a", 1.0), {"a"}),
        ((0.0, "b"), {"b"}),
        (("c", "d"), {"c", "d"}),
        (("a", "a"), {"a"}),  # duplicates collapsed by set semantics
    ],
)
def test_parse_bounds_extracts_dependencies(bounds, expected):
    assert utils.parse_bounds(bounds) == expected


def test_build_dependency_graph_simple_chain():
    param_dict = {"a": (0, 1), "b": ("a", 2), "c": ("b", "a")}
    graph = utils.build_dependency_graph(param_dict)
    assert graph == {"a": {"b", "c"}, "b": {"c"}, "c": set()}


def test_build_dependency_graph_includes_dependency_only_params():
    param_dict = {"a": (0, 1), "b": ("c", 2), "c": (0, 1)}
    graph = utils.build_dependency_graph(param_dict)
    assert graph["c"] == {"b"}
    assert graph["a"] == set()


def test_build_dependency_graph_raises_on_unknown_dependency():
    param_dict = {"a": (0, "unknown")}
    with pytest.raises(ValueError, match="unknown"):
        utils.build_dependency_graph(param_dict)


def test_topological_sort_util_builds_order():
    graph = {"a": {"b"}, "b": set()}
    visited: set[str] = set()
    stack: list[str] = []
    utils.topological_sort_util("a", visited, stack, graph, set())
    assert stack == ["a", "b"]
    assert visited == {"a", "b"}


def test_topological_sort_util_detects_cycle():
    graph = {"a": {"b"}, "b": {"a"}}
    visited: set[str] = set()
    stack: list[str] = []
    with pytest.raises(ValueError, match="Circular dependency"):
        utils.topological_sort_util("a", visited, stack, graph, set())


def test_topological_sort_returns_valid_order():
    graph = {"a": {"b", "c"}, "b": {"d"}, "c": {"d"}, "d": set()}
    order = utils.topological_sort(graph)
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


def test_topological_sort_propagates_cycle_error():
    graph = {"a": {"b"}, "b": {"a"}}
    with pytest.raises(ValueError, match="Circular dependency"):
        utils.topological_sort(graph)


def test_sample_parameters_from_constraints_independent_bounds():
    np.random.seed(0)
    param_dict = {"a": (0.0, 1.0), "b": (-1.0, 1.0)}
    samples = utils.sample_parameters_from_constraints(param_dict, sample_size=5)
    assert set(samples.keys()) == {"a", "b"}
    assert samples["a"].shape == (5,)
    assert samples["b"].shape == (5,)
    assert np.all((samples["a"] >= 0.0) & (samples["a"] <= 1.0))
    assert np.all((samples["b"] >= -1.0) & (samples["b"] <= 1.0))


def test_sample_parameters_from_constraints_with_dependencies():
    np.random.seed(1)
    param_dict = {"a": (0.0, 1.0), "b": ("a", 2.0)}
    samples = utils.sample_parameters_from_constraints(param_dict, sample_size=4)
    assert np.all(samples["b"] >= samples["a"])
    assert np.all(samples["b"] <= 2.0)


def test_sample_parameters_from_constraints_missing_dependency():
    param_dict = {"b": ("a", 2.0)}
    with pytest.raises(ValueError, match="must be defined"):
        utils.sample_parameters_from_constraints(param_dict, sample_size=1)


def test_sample_parameters_from_constraints_invalid_bounds():
    param_dict = {"a": (0.0, 1.0), "b": ("a", "a")}
    with pytest.raises(ValueError, match="must be less than upper bound"):
        utils.sample_parameters_from_constraints(param_dict, sample_size=3)


def test_sample_parameters_from_constraints_independent():
    np.random.seed(0)
    param_dict = {"a": (0.0, 1.0), "b": (-1.0, 1.0)}
    samples = utils.sample_parameters_from_constraints(param_dict, sample_size=5)
    assert set(samples.keys()) == {"a", "b"}
    assert samples["a"].shape == (5,)
    assert samples["b"].shape == (5,)
    assert np.all(samples["a"] >= 0.0)
    assert np.all(samples["a"] <= 1.0)
    assert np.all(samples["b"] >= -1.0)
    assert np.all(samples["b"] <= 1.0)


def test_sample_parameters_from_constraints_with_dependency():
    np.random.seed(1)
    param_dict = {"a": (0.0, 1.0), "b": ("a", 2.0)}
    samples = utils.sample_parameters_from_constraints(param_dict, sample_size=4)
    assert samples["a"].shape == (4,)
    assert samples["b"].shape == (4,)
    assert np.all(samples["b"] >= samples["a"])
    assert np.all(samples["b"] <= 2.0)


def test_sample_parameters_from_constraints_missing_dependency():
    np.random.seed(0)
    param_dict = {"a": ("missing", 1.0)}
    with pytest.raises(ValueError, match="not defined"):
        utils.sample_parameters_from_constraints(param_dict, sample_size=2)


def test_sample_parameters_from_constraints_cycle():
    np.random.seed(0)
    param_dict = {"a": (0.0, "b"), "b": (0.0, "a")}
    with pytest.raises(ValueError, match="Error in topological sorting"):
        utils.sample_parameters_from_constraints(param_dict, sample_size=2)