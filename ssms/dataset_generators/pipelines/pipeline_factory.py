"""Factory for creating data generation pipelines."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ssms.dataset_generators.protocols import DataGenerationPipelineProtocol

from ssms.dataset_generators.pipelines.simulation_pipeline import SimulationPipeline
from ssms.dataset_generators.pipelines.pyddm_pipeline import PyDDMPipeline


def create_data_generation_pipeline(
    generator_config: dict,
    model_config: dict,
) -> "DataGenerationPipelineProtocol":
    """Create appropriate data generation pipeline based on configuration.

    This factory selects the right pipeline based on estimator_type in
    generator_config:
    - 'kde': SimulationPipeline (runs simulations + KDE)
    - 'pyddm': PyDDMPipeline (analytical Fokker-Planck solution)

    The factory internally creates all required components (estimator builder,
    training strategy, parameter sampler) from the provided configurations.
    The model_config serves as the single source of truth for model
    specifications, parameter bounds, and transformations.

    Args:
        generator_config: Configuration dict with 'estimator.type' key
        model_config: Model specification (must contain 'param_bounds_dict')

    Returns:
        Data generation pipeline instance

    Raises:
        ValueError: If estimator_type is unknown

    Note:
        The same model_config can be used for both simulation-based (KDE) and
        analytical (PyDDM) pipelines. Custom boundaries and drift functions
        should be configured through the model_config or using ModelConfigBuilder.

    Example:
        >>> pipeline = create_data_generation_pipeline(
        ...     generator_config=config,
        ...     model_config=model_config,
        ... )
        >>> # Pipeline is ready to use with all components created internally
    """
    # Import component classes (not instances - pipelines will instantiate)
    from ssms.config.config_utils import get_nested_config
    from ssms.dataset_generators.estimator_builders.kde_builder import (
        KDEEstimatorBuilder,
    )
    from ssms.dataset_generators.strategies import ResampleMixtureStrategy

    # Get estimator type from nested config
    estimator_type = get_nested_config(
        generator_config, "estimator", "type", default="kde"
    ).lower()

    if estimator_type == "kde":
        # Pass classes - pipeline will instantiate with its configs
        return SimulationPipeline(
            generator_config=generator_config,
            model_config=model_config,
            estimator_builder=KDEEstimatorBuilder,  # Class, not instance!
            training_strategy=ResampleMixtureStrategy,  # Class, not instance!
        )
    elif estimator_type == "pyddm":
        # For PyDDM, need the specific builder
        from ssms.dataset_generators.estimator_builders.pyddm_builder import (
            PyDDMEstimatorBuilder,
        )

        return PyDDMPipeline(
            generator_config=generator_config,
            model_config=model_config,
            estimator_builder=PyDDMEstimatorBuilder,  # Class, not instance!
            training_strategy=ResampleMixtureStrategy,  # Class, not instance!
        )
    else:
        raise ValueError(
            f"Unknown estimator_type: '{estimator_type}'. "
            f"Supported types: ['kde', 'pyddm']"
        )
