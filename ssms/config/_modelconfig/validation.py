"""Validation helpers for model configuration dicts.

This enforces rules for parameter names and for the optional
``ndt_edge_shift`` declaration.
"""

from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any, List

NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9\.]{0,30}$")

NDT_EDGE_SHIFT_KEYS = {"param", "scale"}


def is_valid_param_name(name: Any) -> bool:
    """Return True if ``name`` is an allowed parameter name.

    Rules:
    - must be a str
    - must match NAME_RE
    """
    if not isinstance(name, str):
        return False
    return bool(NAME_RE.match(name))


def get_invalid_param_names(params: list) -> List[str]:
    """Validate only the parameter names in ``params``.

    This checks that `params` is a list, that there are no duplicates,
    and that every name satisfies `is_valid_param_name`.

    """
    if not isinstance(params, list):
        raise TypeError(f"'params' must be a list, got {type(params).__name__}")
    # duplicates
    seen = set()
    dupes: List[str] = []
    for p in params:
        if p in seen:
            dupes.append(p)
        seen.add(p)
    if dupes:
        raise ValueError(f"Duplicate parameter names: {sorted(set(dupes))}")
    # name checks
    invalid = [p for p in params if not is_valid_param_name(p)]
    return invalid


def get_invalid_configs(configs: dict[str, dict]) -> list[str]:
    return [
        name for name, cfg in configs.items() if get_invalid_param_names(cfg["params"])
    ]


def get_ndt_edge_shift_errors(config: dict) -> List[str]:
    """Validate the optional ``ndt_edge_shift`` entry of ``config``.

    ``{"param": <name>, "scale": s}`` declares that the model's response-time
    support starts at ``t - s * <name>`` rather than at ``t`` (absent means the
    support starts at ``t``). When present it must be a dict with exactly those
    two keys, both ``<name>`` and ``"t"`` must be in ``config["params"]``, and
    ``s`` must be a finite non-negative number (bools are rejected).

    Returns the list of problems found; empty when the key is absent or valid.
    """
    if "ndt_edge_shift" not in config:
        return []
    shift = config["ndt_edge_shift"]
    if not isinstance(shift, dict):
        return [f"ndt_edge_shift must be a dict, got {type(shift).__name__}"]
    errors: List[str] = []
    if set(shift) != NDT_EDGE_SHIFT_KEYS:
        errors.append(
            f"ndt_edge_shift must have exactly the keys "
            f"{sorted(NDT_EDGE_SHIFT_KEYS)}, got {list(shift)}"
        )
    params = config.get("params")
    params = params if isinstance(params, list) else []
    if "t" not in params:
        errors.append("ndt_edge_shift requires 't' in params")
    param = shift.get("param")
    if param not in params:
        errors.append(f"ndt_edge_shift param {param!r} is not in params")
    scale = shift.get("scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, Real)
        or not math.isfinite(scale)
        or scale < 0
    ):
        errors.append(
            f"ndt_edge_shift scale must be a finite non-negative number, got {scale!r}"
        )
    return errors


def get_invalid_ndt_edge_shift_configs(configs: dict[str, dict]) -> list[str]:
    return [name for name, cfg in configs.items() if get_ndt_edge_shift_errors(cfg)]


__all__ = [
    "is_valid_param_name",
    "get_invalid_param_names",
    "get_invalid_configs",
    "get_ndt_edge_shift_errors",
    "get_invalid_ndt_edge_shift_configs",
]
