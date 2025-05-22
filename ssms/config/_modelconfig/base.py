"""Base configurations and utilities for model configs."""

from .boundary_functions import (
    constant,
    angle,
    weibull_cdf,
    generalized_logistic,
    conflict_gamma,
)
from .drift_functions import (
    constant as gamma_drift,
    ds_conflict_drift,
    attend_drift,
    attend_drift_simple,
)

# Boundary configurations
boundary_config = {
    "constant": {
        "fun": constant,
        "params": [],
        "multiplicative": True,
    },
    "angle": {
        "fun": angle,
        "params": ["theta"],
        "multiplicative": False,
    },
    "weibull_cdf": {
        "fun": weibull_cdf,
        "params": ["alpha", "beta"],
        "multiplicative": True,
    },
    "generalized_logistic": {
        "fun": generalized_logistic,
        "params": ["B", "M", "v"],
        "multiplicative": True,
    },
    "conflict_gamma": {
        "fun": conflict_gamma,
        "params": ["theta", "scale", "alpha_gamma", "scale_gamma"],
        "multiplicative": False,
    },
}

# Drift configurations
drift_config = {
    "constant": {
        "fun": constant,
        "params": [],
    },
    "gamma_drift": {
        "fun": gamma_drift,
        "params": ["shape", "scale", "c"],
    },
    "ds_conflict_drift": {
        "fun": ds_conflict_drift,
        "params": ["tinit", "dinit", "tslope", "dslope", "tfixedp", "tcoh", "dcoh"],
    },
    "attend_drift": {
        "fun": attend_drift,
        "params": ["ptarget", "pouter", "pinner", "r", "sda"],
    },
    "attend_drift_simple": {
        "fun": attend_drift_simple,
        "params": ["ptarget", "pouter", "r", "sda"],
    },
}
