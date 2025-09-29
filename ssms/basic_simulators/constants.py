from typing import Any


DEFAULT_SIM_PARAMS: dict[str, Any] = {
    "max_t": 20.0,
    "n_samples": 2000,
    "n_trials": 1000,
    "delta_t": 0.001,
    "random_state": None,
    "return_option": "full",
    "smooth_unif": False,
}