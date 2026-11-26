"""Model configs for the multiple-threshold correlated Lognormal race (MTLNR).

Reynolds, Kvam, Osth & Heathcote (2020), "Correlated racing evidence
accumulator models". The paper's binary-choice correlated LNR is already the
library's ``lnr2_corr``; this module adds the paper's confidence extension.

Registered configs (statically, in ``ssms/config/_modelconfig/__init__.py``)
------------------------------------------------------------------------
``mtlnr2_conf2``
    Two confidence levels (one confidence threshold per accumulator), the
    design the paper recommends (Fig. 5: correlation recovered as well as with
    three levels). 4 possible choices.
``mtlnr2_conf3``
    Three confidence levels (low / medium / high; two thresholds per
    accumulator), the model fit to Ratcliff et al.'s (1994) data (Table 2).
    6 possible choices.

Parameters
----------
mu0, mu1, sigma0, sigma1, rho, t
    Exactly as in ``lnr2_corr`` (``rho`` bounds widened to +-0.999 because the
    paper's per-participant estimates reach 0.97-1.00, Table 2).
d1acc{i}, d2acc{i}
    Confidence thresholds of accumulator ``i`` (used when ``i`` loses) as a
    fraction of its choice threshold, in [0, 1]; ``d1 < d2``. Loser below
    ``d1`` -> highest confidence. The paper's ``d1New``/``d2New``/``d1Old``/
    ``d2Old`` with accumulator 0 = "new", 1 = "old". At ``d = 0`` a threshold
    is never crossed from above, i.e. the mechanism is switched off.

Choice coding: ``choice = winner + 2 * conf``; ``winner = choice % 2`` is the
``lnr2_corr`` choice, ``conf = choice // 2`` the confidence level (0 lowest).
"""

from __future__ import annotations

import itertools

from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators.mtlnr import multiple_threshold_lognormal_race
from ssms.transforms.sampling import SwapIfLessConstraint

_MU_BOUNDS = (-3.0, 2.0)
_SIGMA_BOUNDS = (0.1, 2.0)
_RHO_BOUNDS = (-0.999, 0.999)
_D_BOUNDS = (0.0, 1.0)
_T_BOUNDS = (0.0, 2.0)

# lnr2_corr defaults, plus thresholds near the paper's group means (Table 2).
_DEFAULTS = {"mu0": -1.0, "mu1": -0.5, "sigma0": 0.5, "sigma1": 0.7, "rho": 0.0}
_D_DEFAULTS = {1: [0.5], 2: [0.5, 0.8]}
_T_DEFAULT = 0.3


def _threshold_names(n_thr: int) -> list[list[str]]:
    return [[f"d{k}acc{i}" for k in range(1, n_thr + 1)] for i in range(2)]


def get_mtlnr2_config(n_levels: int) -> dict:
    """Config for the MTLNR with ``n_levels`` (2 or 3) confidence levels."""
    if n_levels not in (2, 3):
        raise ValueError("Registered MTLNR configs have 2 or 3 confidence levels.")
    n_thr = n_levels - 1
    params = ["mu0", "mu1", "sigma0", "sigma1", "rho"]
    lower = [_MU_BOUNDS[0]] * 2 + [_SIGMA_BOUNDS[0]] * 2 + [_RHO_BOUNDS[0]]
    upper = [_MU_BOUNDS[1]] * 2 + [_SIGMA_BOUNDS[1]] * 2 + [_RHO_BOUNDS[1]]
    defaults = [_DEFAULTS[p] for p in params]
    sampling = []
    for names in _threshold_names(n_thr):
        params += names
        lower += [_D_BOUNDS[0]] * n_thr
        upper += [_D_BOUNDS[1]] * n_thr
        defaults += list(_D_DEFAULTS[n_thr])
        for lo, hi in itertools.pairwise(names):
            sampling.append(SwapIfLessConstraint(hi, lo))  # ensures hi > lo
    params.append("t")
    lower.append(_T_BOUNDS[0])
    upper.append(_T_BOUNDS[1])
    defaults.append(_T_DEFAULT)

    n_choices = 2 * n_levels
    cfg = {
        "name": f"mtlnr2_conf{n_levels}",
        "params": params,
        "param_bounds": [lower, upper],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": len(params),
        "default_params": defaults,
        "nchoices": n_choices,
        "choices": list(range(n_choices)),
        "n_particles": 2,
        "tags": ["python_simulator"],
        "simulator": multiple_threshold_lognormal_race,
        "parameter_transforms": {"sampling": sampling, "simulation": []},
    }
    # Imported lazily: ``ssms.config._modelconfig`` imports this module.
    from ssms.config._modelconfig import _normalize_param_bounds

    return _normalize_param_bounds(cfg)


def get_mtlnr2_conf2_config() -> dict:
    return get_mtlnr2_config(2)


def get_mtlnr2_conf3_config() -> dict:
    return get_mtlnr2_config(3)


CONFIG_FACTORIES = {
    "mtlnr2_conf2": get_mtlnr2_conf2_config,
    "mtlnr2_conf3": get_mtlnr2_conf3_config,
}
