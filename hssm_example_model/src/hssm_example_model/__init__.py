"""A minimal ssm-configs plugin.

Installing this distribution registers the model ``example_model`` with
``ssm_configs.hssm_registry``. No code has to import this package for that to
happen -- ``ssm_configs`` finds it through the ``ssm_configs`` entry point
declared in ``pyproject.toml``, and imports it itself during discovery.
"""

from pathlib import Path

import pluggy

hookimpl = pluggy.HookimplMarker("ssm_configs")


@hookimpl
def ssm_configs_config_path() -> Path:
    """Return the path to this plugin's model configuration.

    Returns
    -------
    Path
        Path to the JSON config shipped alongside this module.
    """
    return Path(__file__).parent / "example_model.json"
