"""Base class for parameter samplers with dependency resolution."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any
import numpy as np


class AbstractParameterSampler(ABC):
    """Base class for parameter samplers with dependency resolution.

    This class handles:
    - Building dependency graphs from parameter bounds
    - Topological sorting to determine sampling order
    - Applying transforms after sampling
    - Validating parameter spaces

    Subclasses must implement _sample_parameter() to define the specific
    sampling strategy (uniform, Sobol, Latin Hypercube, etc.).
    """

    def __init__(
        self,
        param_space: dict[str, tuple[Any, Any]],
        transforms: list | None = None,
    ):
        """Initialize the parameter sampler.

        Args:
            param_space: Dictionary mapping parameter names to (lower, upper) bounds.
                        Bounds can be numeric or strings (for dependencies).
            transforms: List of transform objects (must have apply() method).
                       Applied after sampling in the order provided.
        """
        self.param_space = param_space
        self.transforms = transforms or []

        # Build dependency graph and sampling order once at initialization
        self._dependency_graph = self._build_dependency_graph()
        self._sampling_order = self._topological_sort()

    @abstractmethod
    def _sample_parameter(
        self,
        param: str,
        lower: float | np.ndarray,
        upper: float | np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample a single parameter (strategy-specific).

        Args:
            param: Name of the parameter being sampled
            lower: Lower bound (scalar or array if dependent on previous samples)
            upper: Upper bound (scalar or array if dependent on previous samples)
            n_samples: Number of samples to generate
            rng: Random number generator to use for sampling

        Returns:
            Array of sampled values (length n_samples)
        """
        ...

    def sample(
        self,
        n_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample parameters respecting dependencies and applying transforms.

        Args:
            n_samples: Number of parameter sets to sample
            rng: Random number generator. If None, uses np.random.default_rng()

        Returns:
            Dictionary mapping parameter names to sampled arrays

        Raises:
            ValueError: If parameter dependencies cannot be resolved
        """
        # Use provided RNG or create a new one
        if rng is None:
            rng = np.random.default_rng()

        samples = {}

        # Sample in topological order (dependencies first)
        for param in self._sampling_order:
            if param not in self.param_space:
                continue

            lower, upper = self.param_space[param]

            # Resolve dependencies (e.g., st depends on t)
            if isinstance(lower, str):
                if lower not in samples:
                    raise ValueError(
                        f"Parameter '{lower}' must be sampled before '{param}'."
                    )
                lower = samples[lower]
            if isinstance(upper, str):
                if upper not in samples:
                    raise ValueError(
                        f"Parameter '{upper}' must be sampled before '{param}'."
                    )
                upper = samples[upper]

            # Sample using strategy-specific method with provided RNG
            samples[param] = self._sample_parameter(param, lower, upper, n_samples, rng)

        # Apply transforms (each transform modifies the dict in place or returns modified dict)
        for transform in self.transforms:
            samples = transform.apply(samples)

        return samples

    def _build_dependency_graph(self) -> dict[str, set[str]]:
        """Build dependency graph from parameter bounds.

        The graph is structured as parent -> children, where children depend on parent.
        For example, if 'st' has upper bound 't', then graph['t'] contains 'st'.

        Returns:
            Dictionary where keys are parameters and values are sets of parameters
            that depend on them.

        Raises:
            ValueError: If a dependency references an undefined parameter
        """
        graph: dict[str, set[str]] = defaultdict(set)
        all_params = set(self.param_space.keys())

        for param, bounds in self.param_space.items():
            # Extract dependencies from bounds
            dependencies = set()
            for value in bounds:
                if isinstance(value, str):
                    dependencies.add(value)

            # Validate dependencies exist
            for dependency in dependencies:
                if dependency not in all_params:
                    raise ValueError(
                        f"Parameter '{param}' depends on '{dependency}', "
                        f"but '{dependency}' is not defined in param_space."
                    )
                # Add edge: dependency -> param (param depends on dependency)
                graph[dependency].add(param)

            all_params.update(dependencies)

        # Ensure all parameters are in the graph (even those with no dependents)
        for param in all_params:
            if param not in graph:
                graph[param] = set()

        return dict(graph)

    def _topological_sort(self) -> list[str]:
        """Perform topological sort to determine sampling order.

        Uses depth-first search to create a valid sampling order where all
        dependencies are sampled before the parameters that depend on them.

        Returns:
            List of parameter names in sampling order

        Raises:
            ValueError: If circular dependencies are detected
        """
        visited: set[str] = set()
        temp_marks: set[str] = set()
        stack: list[str] = []

        def visit(node: str):
            """DFS helper for topological sort."""
            if node in temp_marks:
                raise ValueError(
                    f"Circular dependency detected involving parameter '{node}'"
                )
            if node in visited:
                return

            temp_marks.add(node)
            # Visit all parameters that depend on this one
            for neighbor in self._dependency_graph.get(node, set()):
                visit(neighbor)
            temp_marks.remove(node)
            visited.add(node)
            # Prepend to ensure dependencies come first
            stack.insert(0, node)

        # Visit all nodes
        for node in self._dependency_graph:
            if node not in visited:
                visit(node)

        return stack

    def get_param_space(self) -> dict[str, tuple[Any, Any]]:
        """Get the parameter space bounds.

        Returns:
            Dictionary mapping parameter names to (lower, upper) bound tuples
        """
        return self.param_space
