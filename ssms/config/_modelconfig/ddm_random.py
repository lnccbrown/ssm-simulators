"""Configuration for DDM models with random variables."""

import functools
import numpy as np
import scipy.stats as sps

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.transforms import ApplyMapping, LambdaAdaptation


def get_ddm_st_config():
    """Get configuration for DDM with random non-decision time."""
    return {
        "name": "ddm_st",
        "params": ["v", "a", "z", "t", "st"],
        "param_bounds": [
            [-3.0, 0.3, 0.3, 0.25, 1e-3],
            [3.0, 2.5, 0.7, 2.25, 0.25],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 0.25, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm_rv,
        "simulator_fixed_params": {
            "z_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
            "v_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
        },
        "simulator_param_mappings": {
            "t_dist": lambda st: functools.partial(
                sps.uniform.rvs, loc=(-1) * st, scale=2 * st
            ),
        },
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "z_dist": cfg["simulator_fixed_params"]["z_dist"],
                                "v_dist": cfg["simulator_fixed_params"]["v_dist"],
                            }
                        )
                        or theta
                    ),
                    name="set_fixed_params",
                ),
                ApplyMapping("st", "t_dist", "t_dist"),
            ],
        },
    }


def get_ddm_uniform_st_config():
    """Get configuration for DDM with uniform non-decision time (alias of ddm_st).

    Description
    -----------
    ``ddm_uniform_st`` is ``ddm_st`` under an explicit name: the returned
    configuration is identical in every field but ``name``. The uniform kernel
    draws ndt ~ Uniform(t - st, t + st), so ``st`` is a HALF-WIDTH
    (SD = st / sqrt(3)), against the STANDARD DEVIATION that ``ddm_normal_st``
    means by the same parameter. Naming the kernel in the model name keeps that
    distinction visible at the call site. ``ddm_st`` remains registered and
    unchanged; this is an addition, not a rename.

    See Also
    --------
    get_ddm_st_config : the same configuration under its original name.
    get_ddm_normal_st_config : same parameters, untruncated Normal ndt kernel.
    """
    return {**get_ddm_st_config(), "name": "ddm_uniform_st"}


def get_ddm_normal_st_config():
    """Get configuration for DDM with untruncated-normal non-decision time.

    Description
    -----------
    Trial-wise non-decision time is drawn from an untruncated Normal kernel,
    ndt ~ Normal(t, st), so ``st`` is the kernel's STANDARD DEVIATION. That is
    the contrast with ``ddm_st``, which exposes the same five parameters over
    the same bounds but draws ndt ~ Uniform(t - st, t + st), where ``st`` is a
    half-width (SD = st / sqrt(3)). The two are not interchangeable at equal
    ``st``: equal dispersion needs st_uniform = sqrt(3) * st_normal.

    Parameters
    ----------
    v : float
        Drift rate. Valid range: [-3.0, 3.0]
    a : float
        Boundary separation. Valid range: [0.3, 2.5]
    z : float
        Starting point as a fraction of ``a``. Valid range: [0.3, 0.7]
    t : float
        Mean non-decision time, in seconds. Valid range: [0.25, 2.25]
    st : float
        Standard deviation of the non-decision-time kernel, in seconds.
        Valid range: [1e-3, 0.25]

    Model Characteristics
    ---------------------
    - Number of choices: 2
    - Boundary type: constant
    - Drift type: constant
    - Key assumption: the ndt kernel is Normal and untruncated.

    Notes
    -----
    - The kernel's support is unbounded, so ndt has no floor and can fall below
      0 near the lower ``t`` bound, yielding a non-positive RT. Positive RTs
      are therefore never guaranteed, anywhere in the box - only made rare, at
      a rate that falls off steeply in t / st: measured 8.9% at the corner
      t=0.25, st=0.25, a=0.3, 0.013% on the 3-sigma edge 3 * st = t
      (st = 0.0833 at t = 0.25), and none in 6e5 draws at st = 0.05. Consumers
      that cannot represent a non-positive RT (KDE-based likelihood estimation
      in particular) must filter, not assume. ``ddm_st``'s bounded kernel
      cannot produce one anywhere in its box.
    - ``param_bounds`` are deliberately identical to ``ddm_st``'s so the two
      models span the same box; the ``st`` bounds therefore carry different
      meanings in the two models.

    See Also
    --------
    get_ddm_st_config : same parameters, uniform (bounded) ndt kernel.
    get_ddm_truncnormt_config : Normal ndt kernel truncated at 0.
    """
    return {
        "name": "ddm_normal_st",
        "params": ["v", "a", "z", "t", "st"],
        "param_bounds": [
            [-3.0, 0.3, 0.3, 0.25, 1e-3],
            [3.0, 2.5, 0.7, 2.25, 0.25],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 0.25, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm_rv,
        "simulator_fixed_params": {
            "z_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
            "v_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
        },
        "simulator_param_mappings": {
            "t_dist": lambda st: functools.partial(sps.norm.rvs, loc=0, scale=st),
        },
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "z_dist": cfg["simulator_fixed_params"]["z_dist"],
                                "v_dist": cfg["simulator_fixed_params"]["v_dist"],
                            }
                        )
                        or theta
                    ),
                    name="set_fixed_params",
                ),
                ApplyMapping("st", "t_dist", "t_dist"),
            ],
        },
    }


def get_ddm_truncnormt_config():
    """Get configuration for DDM with truncated normal non-decision time."""
    return {
        "name": "ddm_truncnormt",
        "params": ["v", "a", "z", "mt", "st"],
        "param_bounds": [
            [-3.0, 0.3, 0.3, 0.05, 1e-3],
            [3.0, 2.5, 0.7, 2.25, 0.5],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 0.25, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm_rv,
        "simulator_fixed_params": {
            "z_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
            "v_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
            "t": 0.0,
        },
        "simulator_param_mappings": {
            "t_dist": lambda mt, st: functools.partial(
                sps.truncnorm.rvs,
                a=(-1) * np.divide(mt, st),
                b=np.inf,
                loc=mt,
                scale=st,
            ),
        },
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "z_dist": cfg["simulator_fixed_params"]["z_dist"],
                                "v_dist": cfg["simulator_fixed_params"]["v_dist"],
                                "t": np.array([0], dtype=np.float32),
                            }
                        )
                        or theta
                    ),
                    name="set_fixed_params_and_zero_t",
                ),
                ApplyMapping("mt", "t_dist", "t_dist", additional_sources=["st"]),
            ],
        },
    }


def get_ddm_rayleight_config():
    """Get configuration for DDM with Rayleigh non-decision time."""
    return {
        "name": "ddm_rayleight",
        "params": ["v", "a", "z", "st"],
        "param_bounds": [
            [-3.0, 0.3, 0.3, 1e-3],
            [3.0, 2.5, 0.7, 1.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 4,
        "default_params": [0.0, 1.0, 0.5, 0.2],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm_rv,
        "simulator_fixed_params": {
            "z_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
            "v_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
            "t": 0.0,
        },
        "simulator_param_mappings": {
            "t_dist": lambda st: functools.partial(
                sps.rayleigh.rvs,
                loc=0,
                scale=st,
            ),
        },
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "z_dist": cfg["simulator_fixed_params"]["z_dist"],
                                "v_dist": cfg["simulator_fixed_params"]["v_dist"],
                                "t": (
                                    np.ones(n) * cfg["simulator_fixed_params"]["t"]
                                ).astype(np.float32),
                            }
                        )
                        or theta
                    ),
                    name="set_fixed_params_and_t",
                ),
                ApplyMapping("st", "t_dist", "t_dist"),
            ],
        },
    }


def get_ddm_sdv_config():
    """Get configuration for DDM with random drift rate."""
    return {
        "name": "ddm_sdv",
        "params": ["v", "a", "z", "t", "sv"],
        "param_bounds": [[-3.0, 0.3, 0.1, 1e-3, 1e-3], [3.0, 2.5, 0.9, 2.0, 2.5]],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 5,
        "default_params": [0.0, 1.0, 0.5, 1e-3, 1e-3],
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm_rv,
        "simulator_fixed_params": {
            "z_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
            "t_dist": functools.partial(sps.norm.rvs, loc=0, scale=0),
        },
        "simulator_param_mappings": {
            "v_dist": lambda sv: functools.partial(
                sps.norm.rvs,
                loc=0,
                scale=sv,
            ),
        },
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "z_dist": cfg["simulator_fixed_params"]["z_dist"],
                                "t_dist": cfg["simulator_fixed_params"]["t_dist"],
                            }
                        )
                        or theta
                    ),
                    name="set_fixed_dists",
                ),
                ApplyMapping("sv", "v_dist", "v_dist"),
            ],
        },
    }
