"""Configuration dictionary for simulators.

Variables:
---------
model_config: dict
    Dictionary containing all the information about the models
"""

from ._modelconfig import get_model_config


def boundary_config_to_function_params(config: dict) -> dict:
    """
    Convert boundary configuration to function parameters.

    Parameters
    ----------
    config: dict
        Dictionary containing the boundary configuration

    Returns
    -------
    dict
        Dictionary with adjusted key names so that they match function parameters names directly.
    """
    return {f"boundary_{k}": v for k, v in config.items()}


model_config = get_model_config()
