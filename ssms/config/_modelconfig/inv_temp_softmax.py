"""Inverse-temperature softmax choice-only model configurations."""

from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators.inv_temp_softmax import inv_temp_softmax


def _get_inv_temp_softmax_config(n_choices: int) -> dict:
    """Return a choice-only softmax config for ``n_choices`` alternatives."""
    q_params = [f"q{i}" for i in range(n_choices)]
    params = ["beta", *q_params]
    return {
        "name": f"inv_temp_softmax_{n_choices}",
        "params": params,
        "param_bounds": [
            [0.0, *([0.0] * n_choices)],
            [10.0, *([1.0] * n_choices)],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": len(params),
        "default_params": [1.0, *([0.5] * n_choices)],
        "nchoices": n_choices,
        "choices": list(range(n_choices)),
        "n_particles": 1,
        "tags": ["choice_only_rl"],
        "simulator": inv_temp_softmax,
    }


def get_inv_temp_softmax_2_config() -> dict:
    """Get configuration for two-choice inverse-temperature softmax."""
    return _get_inv_temp_softmax_config(n_choices=2)


def get_inv_temp_softmax_3_config() -> dict:
    """Get configuration for three-choice inverse-temperature softmax."""
    return _get_inv_temp_softmax_config(n_choices=3)
