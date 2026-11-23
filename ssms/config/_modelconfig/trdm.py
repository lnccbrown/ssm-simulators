"""Timed Racing Diffusion Model (TRDM) configurations.

Description
-----------
Hawkins & Heathcote's (2021) TRDM races ``N`` one-boundary (Wald) evidence
accumulators, one per response, against one one-boundary *timing*
accumulator. An evidence win makes that response; a timer win makes an
unbiased guess (probability ``1/N`` per response). No parameter varies from
trial to trial.

Three configurations are registered:

``trdm_2``
    The paper's two-choice TRDM. Each accumulator has its own drift and its own
    threshold (the paper's mean/sensitivity and mean/bias parameterisations,
    ``gamma_{E,1|2} = gamma_mu +/- gamma_zeta/2`` and
    ``alpha_{E,1|2} = alpha_mu +/- alpha_zeta/2``, are re-expressed per
    accumulator).
``trdm_2_guess``
    ``trdm_2`` with a free timer guess probability ``g`` (probability that a
    timer-terminated trial gives response 0; the paper's ``g_{T,1}`` in
    Eq. 1). The paper's Part 2 guessing variants (free, design-matched
    ``g = .75``, threshold-derived ``g = 1 - b0/(b0 + b1)``) and its go/no-go
    application (timer responses are "go") are settings of ``g``. At
    ``g = 0.5`` it is ``trdm_2``, sample for sample at a given seed.
``trdm_3``
    ``racing_diffusion_3`` plus a timer: three accumulators with a shared
    threshold ``b`` and the library RDM's Uniform(0, A) start point. With the
    timer switched off (``vT = 0``, ``sT = 0``) it is ``racing_diffusion_3``
    (same parameters, same units). The paper's TRDM is ``A = 0``.

Parameters (library units)
--------------------------
v0, v1, (v2) : evidence drift per accumulator, in units of the evidence noise
    ``s`` (``sigma_noise``, default 1): ``v_i = gamma_{E,i} / sigma_E``.
    ``trdm_2`` allows negative drifts (the paper's ``gamma_mu - gamma_zeta/2``
    is negative in several fits); such an accumulator may never finish.
b0, b1 (trdm_2) / b (trdm_3) : evidence threshold(s), ``alpha_{E,i} / sigma_E``.
A (trdm_3) : Uniform(0, A) evidence start point, as in ``racing_diffusion_3``.
t : evidence onset / non-decision time ``tau_E`` (s).
vT : timer drift ``gamma_T`` (timer threshold fixed at 1, paper units).
sT : timer diffusion coefficient ``sigma_T`` (paper units).
g (trdm_2_guess) : probability that a timer-terminated trial is answered
    with response 0; 0.5 is the paper's unbiased guess.
tT : timer onset ``tau_T`` (s); the paper fixes it at 0.05 (0 for the go/no-go
    data).

Switching the timer off: ``vT = 0`` and ``sT = 0`` (the timer never reaches
threshold). ``sT = 0`` with ``vT > 0`` is a hard internal deadline at
``tT + 1/vT``.

Notes
-----
- Response times can be shorter than ``t``: the timer can finish before the
  evidence onset when ``tT < t``. The RT support starts at ``min(t, tT)``,
  which ``ndt_edge_shift`` cannot express; KDE training with
  ``displace_t=True`` is not valid for these models.
- ``delta_t``/``smooth_unif`` are ignored; finishing times are exact.
"""

from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators.trdm import timed_racing_diffusion

# Bounds bracket the paper's estimates (Tables 3, 4, 6, 7, 11, 12) after the
# conversion v = gamma / sigma_E, b = alpha / sigma_E: e.g. Table 11 go|go
# 4.63/.72 = 6.4, Table 7 error accumulator (.83 - 4.22/2)/.49 = -2.6,
# Table 7 threshold 1/.49 = 2.04, Table 12 accuracy 1.61/.94 = 1.71.
_V2_BOUNDS = (-3.0, 7.0)
_B2_BOUNDS = (0.2, 4.0)
_T_BOUNDS = (0.0, 2.0)
_VT_BOUNDS = (0.0, 5.0)
_ST_BOUNDS = (0.0, 2.0)
_TT_BOUNDS = (0.0, 1.0)


def get_trdm_2_config() -> dict:
    """Two-choice TRDM (the model fitted throughout the paper)."""
    params = ["v0", "v1", "b0", "b1", "t", "vT", "sT", "tT"]
    lower = [
        _V2_BOUNDS[0],
        _V2_BOUNDS[0],
        _B2_BOUNDS[0],
        _B2_BOUNDS[0],
        _T_BOUNDS[0],
        _VT_BOUNDS[0],
        _ST_BOUNDS[0],
        _TT_BOUNDS[0],
    ]
    upper = [
        _V2_BOUNDS[1],
        _V2_BOUNDS[1],
        _B2_BOUNDS[1],
        _B2_BOUNDS[1],
        _T_BOUNDS[1],
        _VT_BOUNDS[1],
        _ST_BOUNDS[1],
        _TT_BOUNDS[1],
    ]
    return {
        "name": "trdm_2",
        "params": params,
        "param_bounds": [lower, upper],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": len(params),
        "default_params": [2.0, 0.5, 1.0, 1.0, 0.3, 1.0, 0.5, 0.05],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "tags": ["python_simulator"],
        "simulator": timed_racing_diffusion,
        "parameter_transforms": {"sampling": [], "simulation": []},
    }


def get_trdm_2_guess_config() -> dict:
    """``trdm_2`` with a free timer guess probability ``g``."""
    cfg = get_trdm_2_config()
    cfg["name"] = "trdm_2_guess"
    cfg["params"] = cfg["params"] + ["g"]
    cfg["param_bounds"] = [
        cfg["param_bounds"][0] + [0.0],
        cfg["param_bounds"][1] + [1.0],
    ]
    cfg["n_params"] = len(cfg["params"])
    cfg["default_params"] = cfg["default_params"] + [0.5]
    return cfg


def get_trdm_3_config() -> dict:
    """``racing_diffusion_3`` plus the TRDM timer."""
    params = ["v0", "v1", "v2", "A", "b", "t", "vT", "sT", "tT"]
    # Shared parameters keep racing_diffusion_3's bounds exactly.
    lower = [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, _VT_BOUNDS[0], _ST_BOUNDS[0], _TT_BOUNDS[0]]
    upper = [2.5, 2.5, 2.5, 1.0, 3.0, 2.0, _VT_BOUNDS[1], _ST_BOUNDS[1], _TT_BOUNDS[1]]
    return {
        "name": "trdm_3",
        "params": params,
        "param_bounds": [lower, upper],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": len(params),
        "default_params": [1.0, 1.0, 1.0, 0.5, 1.5, 1e-3, 0.5, 0.3, 0.0],
        "nchoices": 3,
        "choices": [0, 1, 2],
        "n_particles": 3,
        "tags": ["python_simulator"],
        "simulator": timed_racing_diffusion,
        "parameter_transforms": {"sampling": [], "simulation": []},
    }


CONFIG_FACTORIES = {
    "trdm_2": get_trdm_2_config,
    "trdm_2_guess": get_trdm_2_guess_config,
    "trdm_3": get_trdm_3_config,
}
