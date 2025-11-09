"""Validation helpers for model configuration dicts.

This enforces rules for parameter names.
"""

from __future__ import annotations

import re
from typing import Any, List

NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9\.]{0,30}$")


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


__all__ = ["is_valid_param_name", "get_invalid_param_names", "get_invalid_configs"]
