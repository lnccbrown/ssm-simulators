"""
Registry system for mapping models to parameter adaptations.

The registry provides a centralized way to associate model names with their
required parameter adaptations. It supports both exact model matching and
pattern-based family matching.
"""

from collections.abc import Callable

from .base import ParameterAdaptation


class ParameterAdapterRegistry:
    """Registry mapping models to parameter adaptation pipelines.

    The registry supports two types of lookups:
    1. Exact model match: Direct lookup by model name
    2. Family match: Pattern-based matching using lambda functions

    Exact matches take precedence over family matches, allowing specific
    models to override family defaults.

    Examples
    --------
    >>> registry = ParameterAdapterRegistry()
    >>>
    >>> # Register exact model
    >>> registry.register_model("lba2", [
    ...     SetDefaultValue("nact", 2),
    ...     ColumnStackParameters(["v0", "v1"], "v")
    ... ])
    >>>
    >>> # Register model family
    >>> registry.register_family(
    ...     "race_2",
    ...     lambda m: m.startswith("race_") and m.endswith("_2"),
    ...     [ColumnStackParameters(["v0", "v1"], "v")]
    ... )
    >>>
    >>> # Lookup
    >>> adaptations = registry.get_processor("lba2")  # Exact match
    >>> adaptations = registry.get_processor("race_no_bias_2")  # Family match
    """

    def __init__(self):
        """Initialize empty registry."""
        # Exact model name → transformations
        self._processors: dict[str, list[ParameterAdaptation]] = {}

        # Family name → matcher function
        self._family_matchers: dict[str, Callable[[str], bool]] = {}

        # Track registration order for debugging
        self._registration_order: list[tuple[str, str]] = []  # (type, name)

    def register_model(
        self, model_name: str, adaptations: list[ParameterAdaptation]
    ) -> None:
        """Register adaptations for a specific model.

        Parameters
        ----------
        model_name : str
            Exact model name (e.g., "lba2", "ddm", "race_3")
        adaptations : list[ParameterAdaptation]
            List of adaptations to apply for this model

        Examples
        --------
        >>> registry = ParameterAdapterRegistry()
        >>> registry.register_model("lba3", [
        ...     SetDefaultValue("nact", 3),
        ...     ColumnStackParameters(["v0", "v1", "v2"], "v")
        ... ])
        """
        self._processors[model_name] = adaptations
        self._registration_order.append(("model", model_name))

    def register_family(
        self,
        family_name: str,
        matcher: Callable[[str], bool],
        adaptations: list[ParameterAdaptation],
    ) -> None:
        """Register adaptations for a family of models.

        The matcher function determines which models belong to this family.
        It receives a model name and should return True if the model matches.

        Parameters
        ----------
        family_name : str
            Name for this model family (used as key)
        matcher : Callable[[str], bool]
            Function that takes a model name and returns True if it matches
        adaptations : list[ParameterAdaptation]
            List of adaptations to apply for matching models

        Examples
        --------
        >>> registry = ParameterAdapterRegistry()
        >>>
        >>> # Match all race models with 2 choices
        >>> registry.register_family(
        ...     "race_2",
        ...     lambda m: m.startswith("race_") and m.endswith("_2"),
        ...     [ColumnStackParameters(["v0", "v1"], "v")]
        ... )
        >>>
        >>> # Match all models with "no_bias" in the name
        >>> registry.register_family(
        ...     "no_bias_models",
        ...     lambda m: "no_bias" in m,
        ...     [SetDefaultValue("z", 0.5)]
        ... )
        """
        self._family_matchers[family_name] = matcher
        self._processors[family_name] = adaptations
        self._registration_order.append(("family", family_name))

    def get_processor(self, model_name: str) -> list[ParameterAdaptation]:
        """Get adaptation pipeline for a model.

        Lookup priority:
        1. Exact model name match
        2. First matching family
        3. Empty list (no adaptations)

        Parameters
        ----------
        model_name : str
            Name of the model to look up

        Returns
        -------
        list[ParameterAdaptation]
            List of adaptations to apply (empty list if no match)

        Examples
        --------
        >>> adaptations = registry.get_processor("lba2")
        >>> # Returns registered adaptations for "lba2"
        >>>
        >>> adaptations = registry.get_processor("race_no_bias_2")
        >>> # Returns adaptations for "race_2" family (if registered)
        >>>
        >>> adaptations = registry.get_processor("unknown_model")
        >>> # Returns [] (empty list)
        """
        # Try exact match first
        if model_name in self._processors:
            return self._processors[model_name]

        # Try family matchers
        for family_name, matcher in self._family_matchers.items():
            try:
                if matcher(model_name):
                    return self._processors.get(family_name, [])
            except Exception:
                # Matcher raised an exception, skip it
                continue

        # No match found
        return []

    def has_processor(self, model_name: str) -> bool:
        """Check if a processor is registered for a model.

        This checks if the model has a registration (exact or family match),
        even if the transformation list is empty.

        Parameters
        ----------
        model_name : str
            Name of the model to check

        Returns
        -------
        bool
            True if model has a registration (explicit or via family)
        """
        # Check exact match
        if model_name in self._processors:
            return True

        # Check family matchers
        for family_name, matcher in self._family_matchers.items():
            try:
                if matcher(model_name):
                    return True
            except Exception:
                continue

        return False

    def list_registered_models(self) -> list[str]:
        """Get list of explicitly registered model names.

        Returns
        -------
        list[str]
            List of model names with exact registrations
        """
        return [name for typ, name in self._registration_order if typ == "model"]

    def list_registered_families(self) -> list[str]:
        """Get list of registered family names.

        Returns
        -------
        list[str]
            List of family names
        """
        return [name for typ, name in self._registration_order if typ == "family"]

    def describe(self, model_name: str) -> str:
        """Get human-readable description of adaptations for a model.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        str
            Description of the adaptation pipeline

        Examples
        --------
        >>> print(registry.describe("lba3"))
        Model: lba3
        Adaptations:
          1. SetDefaultValue(param_name='nact', default_value=3)
          2. ColumnStackParameters(source_params=['v0', 'v1', 'v2'], ...)
        """
        adaptations = self.get_processor(model_name)

        if not adaptations:
            return f"Model: {model_name}\nNo adaptations registered"

        # Check if exact or family match
        match_type = "exact" if model_name in self._processors else "family"

        lines = [
            f"Model: {model_name} ({match_type} match)",
            f"Adaptations ({len(adaptations)}):",
        ]

        for i, adaptation in enumerate(adaptations, 1):
            lines.append(f"  {i}. {adaptation}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of registry."""
        n_models = len(self.list_registered_models())
        n_families = len(self.list_registered_families())
        return f"ParameterAdapterRegistry({n_models} models, {n_families} families)"


# Global singleton instance - initialized lazily to avoid circular imports
# This is automatically populated with all built-in model adaptations on first access
_GLOBAL_ADAPTER_REGISTRY: ParameterAdapterRegistry | None = None


def _get_or_create_global_registry() -> ParameterAdapterRegistry:
    """Get or create the global registry with all built-in model adaptations.

    This function uses lazy initialization to avoid circular import issues.
    On first call, it imports ModularParameterSimulatorAdapter and uses its
    _build_default_registry() method to get all the built-in registrations.
    This keeps the single source of truth in one place.
    """
    global _GLOBAL_ADAPTER_REGISTRY

    if _GLOBAL_ADAPTER_REGISTRY is None:
        # Import here to avoid circular dependency
        from ssms.basic_simulators.modular_parameter_simulator_adapter import (
            ModularParameterSimulatorAdapter,
        )

        _GLOBAL_ADAPTER_REGISTRY = (
            ModularParameterSimulatorAdapter._build_default_registry()
        )

    return _GLOBAL_ADAPTER_REGISTRY


def register_adapter_to_model(
    model_name: str,
    adaptations: list[ParameterAdaptation],
) -> None:
    """Register parameter adaptations to a specific model globally.

    This associates existing parameter adaptation classes with a model.
    Once registered, the model will use these adaptations automatically when
    simulated with ModularParameterSimulatorAdapter.

    Note: This registers which adaptations a MODEL uses, not a new adapter type.
    To create custom adapter types, subclass ParameterAdaptation directly.

    Parameters
    ----------
    model_name : str
        Unique model name
    adaptations : list[ParameterAdaptation]
        List of parameter adaptation instances to apply for this model

    Examples
    --------
    Register adaptations for a custom model:

    >>> from ssms.basic_simulators.parameter_adapters import register_adapter_to_model
    >>> from ssms.basic_simulators.parameter_adapters import (
    ...     SetDefaultValue, ExpandDimension, ColumnStackParameters
    ... )
    >>>
    >>> register_adapter_to_model("my_model", [
    ...     SetDefaultValue("z", 0.5),
    ...     ExpandDimension(["a", "t"]),
    ...     ColumnStackParameters(["v0", "v1"], "v"),
    ... ])
    >>>
    >>> # Now when simulating with this model, adaptations are applied automatically
    >>> sim = Simulator(model="my_model")

    Override built-in model adaptations:

    >>> # Register custom adaptations for an existing model
    >>> register_adapter_to_model("lba2", [
    ...     ColumnStackParameters(["v0", "v1"], "v"),
    ...     # Custom adaptations here
    ... ])
    """
    registry = _get_or_create_global_registry()
    registry.register_model(model_name, adaptations)


def register_adapter_to_model_family(
    family_name: str,
    matcher: Callable[[str], bool],
    adaptations: list[ParameterAdaptation],
) -> None:
    """Register parameter adaptations to a family of models globally.

    This associates existing parameter adaptation classes with multiple models
    matching a pattern, without having to register each model individually.

    Parameters
    ----------
    family_name : str
        Name for this model family (used as identifier)
    matcher : Callable[[str], bool]
        Function that returns True if a model name matches this family
    adaptations : list[ParameterAdaptation]
        List of parameter adaptation instances to apply for matching models

    Examples
    --------
    Register adaptations for all race models with 2 alternatives:

    >>> from ssms.basic_simulators.parameter_adapters import register_adapter_to_model_family
    >>> from ssms.basic_simulators.parameter_adapters import ColumnStackParameters
    >>>
    >>> register_adapter_to_model_family(
    ...     "race_2",
    ...     lambda m: m.startswith("race_") and m.endswith("_2"),
    ...     [ColumnStackParameters(["v0", "v1"], "v")]
    ... )
    >>>
    >>> # Now all models matching the pattern use these adaptations
    >>> # e.g., "race_no_bias_2", "race_no_z_2", etc.

    Register for models with specific naming pattern:

    >>> register_adapter_to_model_family(
    ...     "no_bias_models",
    ...     lambda m: "no_bias" in m,
    ...     [SetDefaultValue("z", 0.5)]
    ... )
    """
    registry = _get_or_create_global_registry()
    registry.register_family(family_name, matcher, adaptations)


def get_adapter_registry() -> ParameterAdapterRegistry:
    """Get the global parameter adapter registry.

    Use this to access registry methods like list_registered_models() or
    get_processor() directly.

    Returns
    -------
    ParameterAdapterRegistry
        The global parameter adapter registry instance, pre-populated with
        all built-in model adaptations

    Examples
    --------
    >>> from ssms.basic_simulators.parameter_adapters import get_adapter_registry
    >>>
    >>> # List all models with registered adaptations
    >>> registry = get_adapter_registry()
    >>> print(registry.list_registered_models())
    ['lba2', 'lba3', 'lca_3', ...]
    >>>
    >>> # Check if a model has adaptations
    >>> if registry.has_processor("my_model"):
    ...     adaptations = registry.get_processor("my_model")
    >>>
    >>> # Describe adaptations for a model
    >>> print(registry.describe("lba3"))
    """
    return _get_or_create_global_registry()
