"""Base configurations and utilities for model configs."""

from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators import drift_functions as df

# Boundary configurations
boundary_config = {
    "constant": {
        "fun": bf.constant,
        "params": [],
        "multiplicative": True,
    },
    "angle": {
        "fun": bf.angle,
        "params": ["theta"],
        "multiplicative": False,
    },
    "weibull_cdf": {
        "fun": bf.weibull_cdf,
        "params": ["alpha", "beta"],
        "multiplicative": True,
    },
    "generalized_logistic": {
        "fun": bf.generalized_logistic,
        "params": ["B", "M", "v"],
        "multiplicative": True,
    },
    "conflict_gamma": {
        "fun": bf.conflict_gamma,
        "params": ["theta", "scale", "alpha_gamma", "scale_gamma"],
        "multiplicative": False,
    },
}

# Drift configurations
drift_config = {
    "constant": {
        "fun": df.constant,
        "params": [],
    },
    "gamma_drift": {
        "fun": df.gamma_drift,
        "params": ["shape", "scale", "c"],
    },
    "ds_conflict_drift": {
        "fun": df.ds_conflict_drift,
        "params": ["tinit", "dinit", "tslope", "dslope", "tfixedp", "tcoh", "dcoh"],
    },
    "attend_drift": {
        "fun": df.attend_drift,
        "params": ["ptarget", "pouter", "pinner", "r", "sda"],
    },
    "attend_drift_simple": {
        "fun": df.attend_drift_simple,
        "params": ["ptarget", "pouter", "r", "sda"],
    },
}
