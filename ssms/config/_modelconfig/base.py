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
        "params": ["theta", "scale", "alphaGamma", "scaleGamma"],
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
    "conflict_ds_drift": {
        "fun": df.conflict_ds_drift,
        "params": ["tinit", "dinit", "tslope", "dslope", "tfixedp", "tcoh", "dcoh"],
    },
    "conflict_dsstimflex_drift": {
        "fun": df.conflict_dsstimflex_drift,
        "params": [
            "tinit",
            "dinit",
            "tslope",
            "dslope",
            "tfixedp",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
        ],
    },
    "conflict_stimflex_drift": {
        "fun": df.conflict_stimflex_drift,
        "params": ["v_t", "v_d", "tcoh", "dcoh", "tonset", "donset"],
    },
    "conflict_stimflexrel1_drift": {
        "fun": df.conflict_stimflexrel1_drift,
        "params": ["v_t", "v_d", "tcoh", "dcoh", "tonset", "donset"],
    },
    "conflict_stimflexrel1_dual_drift": {
        "fun": df.conflict_stimflexrel1_dual_drift,
        "params": [
            "v_t",
            "v_d",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
            "toffset",
            "doffset",
        ],
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
