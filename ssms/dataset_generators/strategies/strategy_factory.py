"""Factory for creating data generation strategies."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ssms.dataset_generators.protocols import DataGenerationStrategyProtocol

from ssms.dataset_generators.strategies.simulation_based_strategy import (
    SimulationBasedGenerationStrategy,
)
from ssms.dataset_generators.strategies.pyddm_strategy import (
    PyDDMGenerationStrategy,
)


def create_data_generation_strategy(
    generator_config: dict,
    model_config: dict,
) -> "DataGenerationStrategyProtocol":
    """Create appropriate data generation strategy based on configuration.

    This factory selects the right strategy based on estimator_type in
    generator_config:
    - 'kde': SimulationBasedGenerationStrategy (runs simulations + KDE)
    - 'pyddm': PyDDMGenerationStrategy (analytical Fokker-Planck solution)

    The factory internally creates all required components (estimator builder,
    training strategy, parameter sampler) from the provided configurations.
    The model_config serves as the single source of truth for model
    specifications, parameter bounds, and transformations.

    Args:
        generator_config: Configuration dict with 'estimator_type' key
        model_config: Model specification (must contain 'param_bounds_dict')

    Returns:
        Data generation strategy instance

    Raises:
        ValueError: If estimator_type is unknown

    Note:
        The same model_config can be used for both simulation-based (KDE) and
        analytical (PyDDM) strategies. Custom boundaries and drift functions
        should be configured through the model_config or using ConfigBuilder.

    Example:
        >>> strategy = create_data_generation_strategy(
        ...     generator_config=config,
        ...     model_config=model_config,
        ... )
        >>> # Strategy is ready to use with all components created internally
    """
    # Create required components that strategies need
    from ssms.dataset_generators.estimator_builders.builder_factory import (
        create_estimator_builder,
    )
    from ssms.dataset_generators.strategies import ResampleMixtureStrategy

    estimator_builder = create_estimator_builder(generator_config, model_config)
    training_strategy = ResampleMixtureStrategy(generator_config, model_config)

    estimator_type = generator_config.get("estimator_type", "kde").lower()

    # Legacy flag support (backward compatibility)
    if generator_config.get("use_pyddm_pdf", False):
        estimator_type = "pyddm"

    if estimator_type == "kde":
        return SimulationBasedGenerationStrategy(
            generator_config=generator_config,
            model_config=model_config,
            estimator_builder=estimator_builder,
            training_strategy=training_strategy,
        )
    elif estimator_type == "pyddm":
        return PyDDMGenerationStrategy(
            generator_config=generator_config,
            model_config=model_config,
            estimator_builder=estimator_builder,
            training_strategy=training_strategy,
        )
    else:
        raise ValueError(
            f"Unknown estimator_type: '{estimator_type}'. "
            f"Supported types: ['kde', 'pyddm']"
        )
