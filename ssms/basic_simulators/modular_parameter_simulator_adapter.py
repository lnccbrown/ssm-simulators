"""
Modular parameter simulator adapter using adaptation pipelines.

This module provides the ModularParameterSimulatorAdapter class which applies
parameter adaptations defined in model_config['parameter_transforms']['simulation'].
"""

from typing import Any, Protocol


class ParameterSimulatorAdapterProtocol(Protocol):
    """Protocol defining the interface for parameter adapters.

    Any class that implements an `adapt_parameters` method with this signature
    can be used as a parameter adapter.
    """

    def adapt_parameters(
        self, theta: dict[str, Any], model_config: dict[str, Any], n_trials: int
    ) -> dict[str, Any]:
        """Adapt theta parameters for simulator consumption.

        Parameters
        ----------
        theta : dict[str, Any]
            Dictionary of theta parameters
        model_config : dict[str, Any]
            Model configuration dictionary
        n_trials : int
            Number of trials

        Returns
        -------
        dict[str, Any]
            Processed theta parameters
        """
        ...


class ModularParameterSimulatorAdapter:
    """Modular parameter simulator adapter using config-defined transforms.

    This adapter applies a sequence of parameter adaptations defined in the
    model configuration's `parameter_transforms.simulation` field.

    All built-in models define their simulation transforms directly in their
    model config, making this adapter simple and transparent.

    Examples
    --------
    >>> adapter = ModularParameterSimulatorAdapter()
    >>> theta = adapter.adapt_parameters(theta, model_config, n_trials)

    The model_config should have transforms defined like:

    >>> model_config = {
    ...     "name": "lba_angle_3",
    ...     "parameter_transforms": {
    ...         "sampling": [...],
    ...         "simulation": [
    ...             ColumnStackParameters(["v0", "v1", "v2"], "v"),
    ...             ExpandDimension(["a", "z", "theta"]),
    ...             SetZeroArray("t"),
    ...         ],
    ...     },
    ... }
    """

    def adapt_parameters(
        self, theta: dict[str, Any], model_config: dict[str, Any], n_trials: int
    ) -> dict[str, Any]:
        """Process theta by applying config-defined transformations.

        Parameters
        ----------
        theta : dict[str, Any]
            Dictionary of theta parameters
        model_config : dict[str, Any]
            Model configuration dictionary containing `parameter_transforms`
        n_trials : int
            Number of trials

        Returns
        -------
        dict[str, Any]
            Processed theta parameters
        """
        from ssms.config import ModelConfigBuilder

        # Get transforms from model config
        transformations = ModelConfigBuilder.get_simulation_transforms(model_config)

        # Apply transformations in sequence
        for transformation in transformations:
            theta = transformation.apply(theta, model_config, n_trials)

        return theta
