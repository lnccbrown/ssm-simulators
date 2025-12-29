"""Builder factory for creating likelihood estimators based on configuration.

This module provides a factory function that selects the appropriate estimator
builder based on the generator_config settings. This enables seamless switching
between different likelihood estimation methods (KDE, PyDDM, etc.) via
configuration alone.
"""

from typing import Dict, Any, TYPE_CHECKING

from ssms.dataset_generators.estimator_builders.kde_builder import KDEEstimatorBuilder

if TYPE_CHECKING:
    from ssms.dataset_generators.protocols import EstimatorBuilderProtocol


def create_estimator_builder(
    generator_config: Dict[str, Any], model_config: Dict[str, Any]
) -> "EstimatorBuilderProtocol":
    """Create appropriate estimator builder based on configuration.

    This factory function examines the generator_config and returns the
    appropriate builder for likelihood estimation. It supports both explicit
    'estimator_type' specification and legacy 'use_pyddm_pdf' flag.

    Arguments
    ---------
    generator_config : dict
        Configuration dictionary containing:
        - 'estimator_type' (str, optional): Explicit estimator type ('kde' or 'pyddm')
        - 'use_pyddm_pdf' (bool, optional): Legacy flag for PyDDM (if True, uses 'pyddm')
    model_config : dict
        Model configuration dictionary (used for PyDDM builder in Phase 3)

    Returns
    -------
    EstimatorBuilderProtocol
        An estimator builder instance (KDEEstimatorBuilder in Phase 2)

    Raises
    ------
    ImportError
        If 'pyddm' estimator is requested but pyddm package is not installed
    ValueError
        If an unknown estimator_type is specified or if model is incompatible with PyDDM

    Examples
    --------
    >>> # Default to KDE
    >>> builder = create_estimator_builder({}, {})
    >>> isinstance(builder, KDEEstimatorBuilder)
    True

    >>> # Explicit KDE
    >>> config = {"estimator_type": "kde"}
    >>> builder = create_estimator_builder(config, {})
    >>> isinstance(builder, KDEEstimatorBuilder)
    True

    >>> # PyDDM estimator (requires pyddm package)
    >>> config = {"estimator_type": "pyddm"}
    >>> builder = create_estimator_builder(config, model_config["ddm"])
    >>> isinstance(builder, PyDDMEstimatorBuilder)
    True

    Notes
    -----
    PyDDM estimator requires the optional 'pyddm' package. Install with:
    pip install pyddm  or  pip install ssms[pyddm]

    PyDDM is only compatible with single-particle, two-choice, Gaussian noise models.
    Incompatible models will raise ValueError at builder construction.

    See Also
    --------
    KDEEstimatorBuilder : Builder for KDE-based likelihood estimators
    """
    # Check for explicit estimator type from nested config
    from ssms.config.config_utils import get_nested_config

    estimator_type = get_nested_config(
        generator_config, "estimator", "type", default="kde"
    ).lower()

    # Legacy flag support (for backward compatibility)
    if get_nested_config(generator_config, "estimator", "use_pyddm_pdf", default=False):
        estimator_type = "pyddm"

    if estimator_type == "kde":
        return KDEEstimatorBuilder(generator_config)
    elif estimator_type == "pyddm":
        # Phase 3: PyDDM now available!
        try:
            from ssms.dataset_generators.estimator_builders.pyddm_builder import (
                PyDDMEstimatorBuilder,
            )

            return PyDDMEstimatorBuilder(generator_config, model_config)
        except ImportError as e:
            if "pyddm" in str(e).lower():
                raise ImportError(
                    "PyDDM estimator requires 'pyddm' package. "
                    "Install with: pip install pyddm  or  pip install ssms[pyddm]"
                ) from e
            else:
                raise
    else:
        raise ValueError(
            f"Unknown estimator_type: '{estimator_type}'. "
            f"Supported types: ['kde', 'pyddm']"
        )
