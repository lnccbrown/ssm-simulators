"""Base configurations and utilities for model configs."""

from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators import drift_functions as df

# Boundary configurations
boundary_config = {
    "constant": {
        "fun": bf.constant,
        "params": ["a"],
    },
    "angle": {
        "fun": bf.angle,
        "params": ["a", "theta"],
    },
    "weibull_cdf": {
        "fun": bf.weibull_cdf,
        "params": ["a", "alpha", "beta"],
    },
    "generalized_logistic": {
        "fun": bf.generalized_logistic,
        "params": ["a", "B", "M", "v"],
    },
    "conflict_gamma": {
        "fun": bf.conflict_gamma,
        "params": ["a", "theta", "scale", "alphaGamma", "scaleGamma"],
    },
}

# Drift configurations
# Drift functions can accept 'v' as a base drift parameter, mirroring the boundary
# function pattern where 'a' can be included. Most built-in drift functions include 'v'
# for consistency, but it's not strictly required (see conflict_stimflexrel1_dual_drift).
drift_config = {
    "constant": {
        "fun": df.constant,
        "params": ["v"],
    },
    "gamma_drift": {
        "fun": df.gamma_drift,
        "params": ["v", "shape", "scale", "c"],
    },
    "conflict_ds_drift": {
        "fun": df.conflict_ds_drift,
        "params": [
            "v",
            "tinit",
            "dinit",
            "tslope",
            "dslope",
            "tfixedp",
            "tcoh",
            "dcoh",
        ],
    },
    "conflict_dsstimflex_drift": {
        "fun": df.conflict_dsstimflex_drift,
        "params": [
            "v",
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
        "params": [
            "v",
            "vt",
            "vd",
            "tcoh",
            "dcoh",
            "tonset",
            "donset",
        ],
    },
    "conflict_stimflexrel1_drift": {
        "fun": df.conflict_stimflexrel1_drift,
        "params": ["v", "vt", "vd", "tcoh", "dcoh", "tonset", "donset"],
    },
    "conflict_stimflexrel1_dual_drift": {
        "fun": df.conflict_stimflexrel1_dual_drift,
        # Note: v is NOT included here because dual_drift returns 2D array
        # and is handled specially by ddm_flex_leak2
        "params": [
            "vt",
            "vd",
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
        "params": ["v", "ptarget", "pouter", "pinner", "r", "sda"],
    },
    "attend_drift_simple": {
        "fun": df.attend_drift_simple,
        "params": ["v", "ptarget", "pouter", "r", "sda"],
    },
}
