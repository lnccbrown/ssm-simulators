"""
This module provides utility functions for handling parameter dependencies and sampling
parameters within specified constraints.

Functions
---------
parse_bounds(bounds: Tuple[Any, Any]) -> Set[str]
    Parse the bounds of a parameter and extract any dependencies.

build_dependency_graph(param_dict: Dict[str, Tuple[Any, Any]]) -> Dict[str, Set[str]]
    Build a dependency graph based on parameter bounds.

topological_sort_util(node: str,
                      visited: Set[str],
                      stack: List[str],
                      graph: Dict[str, Set[str]],
                      temp_marks: Set[str])
                      -> None
    Helper function for performing a depth-first search in the topological sort.

topological_sort(graph: Dict[str, Set[str]]) -> List[str]
    Perform a topological sort on the dependency graph to determine the sampling order.

sample_parameters_from_constraints(param_dict: Dict[str, Tuple[Any, Any]],
                                   sample_size: int)
                                   -> Dict[str, np.ndarray]
    Sample parameters uniformly within specified bounds, respecting any dependencies.
"""  # noqa: D205, D404

from typing import Any, Dict, List, Set, Tuple
import numpy as np


def parse_bounds(bounds: Tuple[Any, Any]) -> Set[str]:
    """Parse the bounds of a parameter and extract any dependencies.

    Args:
        bounds: A tuple of (lower, upper) bounds. Can be numeric or strings (for dependencies).

    Returns:
        Set of parameter names that this parameter depends on.

    Example:
        >>> parse_bounds((0.0, 1.0))
        set()
        >>> parse_bounds(("a", 1.0))
        {'a'}
        >>> parse_bounds((0.0, "b"))
        {'b'}
        >>> parse_bounds(("c", "d"))
        {'c', 'd'}
    """
    dependencies = set()
    for value in bounds:
        if isinstance(value, str):
            dependencies.add(value)
    return dependencies


def build_dependency_graph(
    param_dict: Dict[str, Tuple[Any, Any]],
) -> Dict[str, Set[str]]:
    """Build a dependency graph based on parameter bounds.

    The graph is structured as parent -> children, where children depend on parent.
    For example, if 'st' has upper bound 't', then graph['t'] contains 'st'.

    Args:
        param_dict: Dictionary mapping parameter names to (lower, upper) bounds.

    Returns:
        Dictionary where keys are parameters and values are sets of parameters
        that depend on them.

    Raises:
        ValueError: If a dependency references an undefined parameter.

    Example:
        >>> param_dict = {"a": (0, 1), "b": ("a", 2)}
        >>> build_dependency_graph(param_dict)
        {'a': {'b'}, 'b': set()}
    """
    graph: Dict[str, Set[str]] = {}
    all_params = set(param_dict.keys())

    # First pass: extract all dependencies
    all_dependencies = set()
    for param, bounds in param_dict.items():
        dependencies = parse_bounds(bounds)
        all_dependencies.update(dependencies)

    # Validate all dependencies exist in param_dict
    for dependency in all_dependencies:
        if dependency not in all_params:
            raise ValueError(
                f"Dependency '{dependency}' is referenced but not defined in param_dict."
            )

    # Build the graph: dependency -> dependents
    for param in all_params:
        graph[param] = set()

    for param, bounds in param_dict.items():
        dependencies = parse_bounds(bounds)
        for dependency in dependencies:
            graph[dependency].add(param)

    return graph


def topological_sort_util(
    node: str,
    visited: Set[str],
    stack: List[str],
    graph: Dict[str, Set[str]],
    temp_marks: Set[str],
) -> None:
    """Helper function for performing a depth-first search in the topological sort.

    Args:
        node: Current node to visit
        visited: Set of permanently visited nodes
        stack: List to accumulate the topological order (prepended in post-order)
        graph: Dependency graph
        temp_marks: Set of temporarily marked nodes (for cycle detection)

    Raises:
        ValueError: If a circular dependency is detected.
    """
    if node in temp_marks:
        raise ValueError(f"Circular dependency detected involving parameter '{node}'")

    if node in visited:
        return

    temp_marks.add(node)

    # Visit all parameters that depend on this one
    for neighbor in graph.get(node, set()):
        topological_sort_util(neighbor, visited, stack, graph, temp_marks)

    temp_marks.remove(node)
    visited.add(node)
    # Prepend to get correct dependency order (dependencies first)
    stack.insert(0, node)


def topological_sort(graph: Dict[str, Set[str]]) -> List[str]:
    """Perform a topological sort on the dependency graph to determine the sampling order.

    Args:
        graph: Dependency graph where keys are parameters and values are sets of
               parameters that depend on them.

    Returns:
        List of parameter names in sampling order (dependencies first).

    Raises:
        ValueError: If circular dependencies are detected.

    Example:
        >>> graph = {"a": {"b"}, "b": set()}
        >>> topological_sort(graph)
        ['a', 'b']
    """
    visited: Set[str] = set()
    temp_marks: Set[str] = set()
    stack: List[str] = []

    for node in graph:
        if node not in visited:
            topological_sort_util(node, visited, stack, graph, temp_marks)

    return stack


def sample_parameters_from_constraints(
    param_dict: Dict[str, Tuple[Any, Any]],
    sample_size: int,
    random_state: int | None = None,
) -> Dict[str, np.ndarray]:
    """Sample parameters uniformly within specified bounds, respecting any dependencies.

    This is a backward-compatible wrapper around UniformParameterSampler.

    Args:
        param_dict: Dictionary mapping parameter names to (lower, upper) bounds.
                   Bounds can be numeric or strings (for dependencies).
        sample_size: Number of parameter sets to sample.
        random_state: Optional random seed for reproducibility.

    Returns:
        Dictionary mapping parameter names to arrays of sampled values.

    Raises:
        ValueError: If bounds are invalid (lower >= upper) or circular dependencies exist.

    Example:
        >>> param_dict = {"v": (-1.0, 1.0), "a": (0.5, 2.0)}
        >>> samples = sample_parameters_from_constraints(param_dict, sample_size=10)
        >>> samples['v'].shape
        (10,)
    """
    from ssms.dataset_generators.parameter_samplers import UniformParameterSampler

    # Validate bounds for parameters with numeric bounds
    for param, (lower, upper) in param_dict.items():
        # Check for same dependency (e.g., ("a", "a"))
        if isinstance(lower, str) and isinstance(upper, str) and lower == upper:
            raise ValueError(
                f"Parameter '{param}' has invalid bounds: lower bound '{lower}' "
                f"must be less than upper bound '{upper}' (same dependency)"
            )
        # Check numeric bounds
        if not isinstance(lower, str) and not isinstance(upper, str):
            if lower >= upper:
                raise ValueError(
                    f"Parameter '{param}' has invalid bounds: lower bound {lower} "
                    f"must be less than upper bound {upper}"
                )

    # Convert random_state to rng if provided
    rng = np.random.default_rng(random_state) if random_state is not None else None

    try:
        sampler = UniformParameterSampler(param_space=param_dict, constraints=[])
        return sampler.sample(n_samples=sample_size, rng=rng)
    except ValueError as e:
        # Re-raise circular dependency errors with backward-compatible message
        if "Circular dependency" in str(e):
            raise ValueError(f"Error in topological sorting: {e}") from e
        # Re-raise other errors as-is
        raise
