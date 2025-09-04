"""
This file contains the expected constrained parameter space for various models.
"""


def infer_constrained_param_space(conf: dict) -> dict:
    """Infer a conservative constrained parameter space from a model config.

    Supports two formats for ``param_bounds``:
    1. ``[[low_1, ...], [high_1, ...]]`` aligned with ``params`` order.
    2. ``{param: (low, high)}`` mapping.

    Silently skips malformed / mismatched entries.
    """
    params = conf.get("params")
    bounds = conf.get("param_bounds")

    if isinstance(bounds, dict):
        return bounds.copy()

    if (
        params
        and isinstance(bounds, (list, tuple))
        and len(bounds) == 2
        and isinstance(bounds[0], (list, tuple))
        and isinstance(bounds[1], (list, tuple))
    ):
        lows, highs = bounds
        return {p: (low, high) for p, low, high in zip(params, lows, highs)}

    return {}
