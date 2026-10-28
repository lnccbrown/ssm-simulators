"""Full DDM model configuration."""

import functools

import scipy.stats as sps  # type: ignore

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.transforms import ApplyMapping


def get_full_ddm_config():
    """Get the configuration for the Full DDM model."""
    return {
        "name": "full_ddm",
        "params": ["v", "a", "z", "t", "sz", "sv", "st"],
        "param_bounds": [
            [-3.0, 0.3, 0.3, 0.25, 1e-3, 1e-3, 1e-3],
            [3.0, 2.5, 0.7, 2.25, 0.2, 2.0, 0.25],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 7,
        "default_params": [0.0, 1.0, 0.5, 0.25, 1e-3, 1e-3, 1e-3],
        # non-decision time is t + U(-st, st), so RTs start at t - st
        "ndt_edge_shift": {"param": "st", "scale": 1.0},
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }


def get_full_ddm_rv_config():
    """Get configuration for full DDM with random variables."""
    return {
        "name": "full_ddm_rv",
        "params": ["v", "a", "z", "t", "sz", "sv", "st"],
        "param_bounds": {
            "v": (-3.0, 3.0),
            "a": (0.3, 2.5),
            "z": (0.3, 0.7),
            "t": (0.25, 2.25),
            "sz": (1e-3, 0.2),
            "sv": (1e-3, 2.0),
            "st": (1e-3, "t"),
        },
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 7,
        "default_params": [0.0, 1.0, 0.5, 0.25, 1e-3, 1e-3, 1e-3],
        # non-decision time is t + U(-st, st), so RTs start at t - st
        "ndt_edge_shift": {"param": "st", "scale": 1.0},
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm_rv,
        "simulator_fixed_params": {},
        "simulator_param_mappings": {
            "t_dist": lambda st: functools.partial(
                sps.uniform.rvs, loc=(-1) * st, scale=2 * st
            ),
            "v_dist": lambda sv: functools.partial(
                sps.norm.rvs,
                loc=0,
                scale=sv,
            ),
            "z_dist": lambda sz: functools.partial(
                sps.uniform.rvs, loc=(-1) * sz, scale=2 * sz
            ),
        },
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                ApplyMapping("sz", "z_dist", "z_dist"),
                ApplyMapping("st", "t_dist", "t_dist"),
                ApplyMapping("sv", "v_dist", "v_dist"),
            ],
        },
    }


def get_full_ddm_lan_config():
    """Get the configuration for the full DDM on the box its LAN was trained on.

    Description
    -----------
    The seven-parameter DDM with trial-to-trial variability in drift (``sv``),
    starting point (``sz``) and non-decision time (``st``), registered under the
    name of the HSSM network trained on it (``full_ddm_lan.onnx``). The box
    below is that network's training box, and HSSM's training-box guard reads
    it from here. The model reaches ``cssm.full_ddm_rv`` through the same
    mapping machinery as ``full_ddm_rv``.

    It is not ``full_ddm`` under another name. The two differ in what ``sz``
    means and in two bounds:

    - ``sz`` is a RELATIVE half-width: the trial's starting point is
      ``z + U(-sz, sz)``, a fraction of the separation, as ``full_ddm_rv``
      draws it. ``full_ddm`` (``cssm.full_ddm``) displaces the starting point by
      ``U(-sz, sz)`` in evidence units, so there ``sz`` depends on ``a`` when
      read as a fraction. HDDM's ``sz`` is the FULL width of this fraction:
      ``sz_hddm = 2 * sz``.
    - ``sv`` is bounded below at 0.3 instead of 1e-3, and above at 2.5 instead
      of 2.0. Below about 0.3 ``sv`` is not identifiable at ordinary trial
      counts, and networks trained down to the floor place spurious density
      there; above 2.0 lies part of the empirical range.
    - ``st`` is bounded above at 0.35 instead of 0.25, the same widening
      ``ddm_st`` and ``ddm_normal_st`` carry (#390).

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
    sz : float
        Half-width of the uniform starting-point kernel, as a fraction of
        ``a``: the trial's starting point is ``z + U(-sz, sz)``.
        Valid range: [1e-3, 0.2]
    sv : float
        Standard deviation of the Normal drift kernel: the trial's drift is
        ``v + Normal(0, sv)``. Valid range: [0.3, 2.5]
    st : float
        Half-width of the uniform non-decision-time kernel, in seconds: the
        trial's non-decision time is ``t + U(-st, st)``, so RTs start at
        ``t - st`` (``ndt_edge_shift``). Valid range: [1e-3, 0.35]

    Model Characteristics
    ---------------------
    - Number of choices: 2
    - Boundary type: constant
    - Drift type: constant
    - Key assumption: all three kernels are independent across trials.

    Notes
    -----
    - ``t`` keeps its lower bound of 0.25 although ``st`` can reach 0.35, so
      in the sliver of the box where ``st > t`` (about 0.7% of a uniform draw
      over it) ``t - st`` is negative and some RTs are non-positive.
      Consumers that cannot represent a non-positive RT must filter.
    - Listed in ``KDE_NO_DISPLACE_T``: RTs start below ``t``, so a KDE fitted
      to its output must not be displaced by ``t``.

    See Also
    --------
    get_full_ddm_config : the same model with ``sz`` in evidence units.
    get_full_ddm_rv_config : the same simulator with ``st`` bounded by ``t``.
    """
    return {
        "name": "full_ddm_lan",
        "params": ["v", "a", "z", "t", "sz", "sv", "st"],
        "param_bounds": [
            [-3.0, 0.3, 0.3, 0.25, 1e-3, 0.3, 1e-3],
            [3.0, 2.5, 0.7, 2.25, 0.2, 2.5, 0.35],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 7,
        "default_params": [0.0, 1.0, 0.5, 0.25, 1e-3, 0.3, 1e-3],
        # non-decision time is t + U(-st, st), so RTs start at t - st
        "ndt_edge_shift": {"param": "st", "scale": 1.0},
        "nchoices": 2,
        "choices": [-1, 1],
        "n_particles": 1,
        "simulator": cssm.full_ddm_rv,
        "simulator_fixed_params": {},
        "simulator_param_mappings": {
            "t_dist": lambda st: functools.partial(
                sps.uniform.rvs, loc=(-1) * st, scale=2 * st
            ),
            "v_dist": lambda sv: functools.partial(sps.norm.rvs, loc=0, scale=sv),
            "z_dist": lambda sz: functools.partial(
                sps.uniform.rvs, loc=(-1) * sz, scale=2 * sz
            ),
        },
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                ApplyMapping("sz", "z_dist", "z_dist"),
                ApplyMapping("st", "t_dist", "t_dist"),
                ApplyMapping("sv", "v_dist", "v_dist"),
            ],
        },
    }
