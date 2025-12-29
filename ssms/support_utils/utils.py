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


# NOTE: Parameter sampling functions have been moved to:
# ssms/dataset_generators/parameter_samplers/
#
# The following functions were removed in the parameter sampling refactor:
# - parse_bounds()
# - build_dependency_graph()
# - topological_sort_util()
# - topological_sort()
# - sample_parameters_from_constraints()
#
# Use UniformParameterSampler from ssms.dataset_generators.parameter_samplers instead.
