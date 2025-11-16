"""
Registry system for mapping models to theta transformations.

The registry provides a centralized way to associate model names with their
required theta transformations. It supports both exact model matching and
pattern-based family matching.
"""

from collections.abc import Callable

from .base import ThetaTransformation


class ThetaProcessorRegistry:
    """Registry mapping models to theta transformation pipelines.

    The registry supports two types of lookups:
    1. Exact model match: Direct lookup by model name
    2. Family match: Pattern-based matching using lambda functions

    Exact matches take precedence over family matches, allowing specific
    models to override family defaults.

    Examples
    --------
    >>> registry = ThetaProcessorRegistry()
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
    >>> transforms = registry.get_processor("lba2")  # Exact match
    >>> transforms = registry.get_processor("race_no_bias_2")  # Family match
    """

    def __init__(self):
        """Initialize empty registry."""
        # Exact model name → transformations
        self._processors: dict[str, list[ThetaTransformation]] = {}

        # Family name → matcher function
        self._family_matchers: dict[str, Callable[[str], bool]] = {}

        # Track registration order for debugging
        self._registration_order: list[tuple[str, str]] = []  # (type, name)

    def register_model(
        self, model_name: str, transformations: list[ThetaTransformation]
    ) -> None:
        """Register transformations for a specific model.

        Parameters
        ----------
        model_name : str
            Exact model name (e.g., "lba2", "ddm", "race_3")
        transformations : list[ThetaTransformation]
            List of transformations to apply for this model

        Examples
        --------
        >>> registry = ThetaProcessorRegistry()
        >>> registry.register_model("lba3", [
        ...     SetDefaultValue("nact", 3),
        ...     ColumnStackParameters(["v0", "v1", "v2"], "v")
        ... ])
        """
        self._processors[model_name] = transformations
        self._registration_order.append(("model", model_name))

    def register_family(
        self,
        family_name: str,
        matcher: Callable[[str], bool],
        transformations: list[ThetaTransformation],
    ) -> None:
        """Register transformations for a family of models.

        The matcher function determines which models belong to this family.
        It receives a model name and should return True if the model matches.

        Parameters
        ----------
        family_name : str
            Name for this model family (used as key)
        matcher : Callable[[str], bool]
            Function that takes a model name and returns True if it matches
        transformations : list[ThetaTransformation]
            List of transformations to apply for matching models

        Examples
        --------
        >>> registry = ThetaProcessorRegistry()
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
        self._processors[family_name] = transformations
        self._registration_order.append(("family", family_name))

    def get_processor(self, model_name: str) -> list[ThetaTransformation]:
        """Get transformation pipeline for a model.

        Lookup priority:
        1. Exact model name match
        2. First matching family
        3. Empty list (no transformations)

        Parameters
        ----------
        model_name : str
            Name of the model to look up

        Returns
        -------
        list[ThetaTransformation]
            List of transformations to apply (empty list if no match)

        Examples
        --------
        >>> transforms = registry.get_processor("lba2")
        >>> # Returns registered transformations for "lba2"
        >>>
        >>> transforms = registry.get_processor("race_no_bias_2")
        >>> # Returns transformations for "race_2" family (if registered)
        >>>
        >>> transforms = registry.get_processor("unknown_model")
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
        """Get human-readable description of transformations for a model.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        str
            Description of the transformation pipeline

        Examples
        --------
        >>> print(registry.describe("lba3"))
        Model: lba3
        Transformations:
          1. SetDefaultValue(param_name='nact', default_value=3)
          2. ColumnStackParameters(source_params=['v0', 'v1', 'v2'], ...)
        """
        transforms = self.get_processor(model_name)

        if not transforms:
            return f"Model: {model_name}\nNo transformations registered"

        # Check if exact or family match
        match_type = "exact" if model_name in self._processors else "family"

        lines = [
            f"Model: {model_name} ({match_type} match)",
            f"Transformations ({len(transforms)}):",
        ]

        for i, transform in enumerate(transforms, 1):
            lines.append(f"  {i}. {transform}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of registry."""
        n_models = len(self.list_registered_models())
        n_families = len(self.list_registered_families())
        return f"ThetaProcessorRegistry({n_models} models, {n_families} families)"
