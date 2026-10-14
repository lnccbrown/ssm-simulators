"""Lognormal race (LNR) model configurations.

Description
-----------
The Lognormal race is the member of Heathcote & Love's (2012) *linear
deterministic accumulation* class in which each accumulator's finishing time is
Lognormal. An accumulator crosses its boundary at ``T = D / V`` (Eq. 1), a
distance over a constant rate with no within-trial noise; if ``D`` and ``V`` are
Lognormal (or either is constant) then so is ``T`` (Eq. 5),

    T_i ~ exp(N(mu_i, sigma_i**2)).

``N`` such accumulators race, the first to finish makes its response, and the
observed RT is ``t + min_i T_i``. Two properties separate the LNR from the
library's other races: Lognormal finishing times are strictly positive, so the
LNR *always* responds (unlike the LBA, whose normal rates can all come out
negative), and because the log finishing times are jointly normal the race stays
tractable when they are correlated across accumulators (Eqs. 8-10), which an
independent race cannot express.

Four configurations are registered, following the library's ``lba2``/``lba3``/
``lba4`` naming for a race family:

``lnr2``, ``lnr3``, ``lnr4``
    Independent Lognormal race with 2, 3 or 4 accumulators. ``lnr2`` is the
    model the paper fits to Wagenmakers et al.'s (2008) lexical-decision data.
``lnr2_corr``
    Two-accumulator race whose log finishing times are correlated (Eqs. 8-10),
    the paper's distinctive extension over the LBA. At ``rho = 0`` it is
    ``lnr2``.

Parameters
----------
mu0, mu1, ... : float
    Mean of ``log T_i`` for accumulator ``i``. Valid range: [-3.0, 2.0].
    Theory: lower ``mu`` means a faster accumulator, i.e. more evidence for
    the response it codes. Distance and rate enter ``mu`` additively
    (``mu = mu_d - mu_v``), which is why the LNR explains speed-accuracy
    trade-off through accumulation rather than through a boundary.
sigma0, sigma1, ... : float
    Standard deviation of ``log T_i``. Valid range: [0.1, 2.0].
    Theory: ``sigma`` is the between-trial variability of the accumulator. A
    larger ``sigma`` on the error accumulator than on the correct one produces
    fast errors; the reverse produces slow errors.
    **Note:** the paper's figures quote the *variance* ``sigma**2``; this
    parameter is its square root.
rho : float
    (``lnr2_corr`` only.) Correlation of ``log T_0`` and ``log T_1``.
    Valid range: [-0.95, 0.95]. Theory: evidence for the two responses is read
    off the same stimulus, so the finishing times need not be independent. The
    paper's fits set this to zero.
t : float
    Shift of the RT distribution (the paper's ``t0``/``theta``: response
    production plus dead time), in seconds. Valid range: [0.0, 2.0].

Model Characteristics
---------------------
- Number of choices: 2 (``lnr2``, ``lnr2_corr``), 3 (``lnr3``), 4 (``lnr4``);
  the accumulator index is the choice, as for ``lba2``/``race_2``.
- Boundary type: constant (the distance to the boundary is folded into
  ``mu``/``sigma``, Eq. 5, so no boundary parameter is exposed).
- Drift type: constant within a trial, variable between trials.
- Key assumptions: no within-trial noise; independent accumulators unless
  ``rho`` is used; constant residual time ``t``; ``sigma > 0`` (the degenerate
  deterministic case is excluded).

References
----------
.. [1] Heathcote, A., & Love, J. (2012). Linear deterministic accumulator
       models of simple choice. *Frontiers in Psychology, 3*, 292.
       https://doi.org/10.3389/fpsyg.2012.00292
.. [2] Brown, S. D., & Heathcote, A. (2008). The simplest complete model of
       choice response time: Linear ballistic accumulation. *Cognitive
       Psychology, 57*(3), 153-178.

Examples
--------
>>> from ssms import Simulator
>>> sim = Simulator("lnr2")
>>> result = sim.simulate(
...     theta={"mu0": -1.2, "mu1": -0.5, "sigma0": 0.447, "sigma1": 0.949,
...            "t": 0.4},
...     n_samples=1000,
...     random_state=0,
... )
>>> import numpy as np
>>> print(f"Mean RT: {np.mean(result['rts']):.3f}")  # doctest: +SKIP
Mean RT: 0.688

See Also
--------
lba2 : Linear ballistic accumulator, the comparison model of the same class
    (uniform distance, normal rate) and the one the paper generalises.
race_2 : Diffusion race, i.e. a race with within-trial noise.
poisson_race : Race between Poisson counters.

Notes
-----
- The simulator draws finishing times directly rather than sampling a distance
  and a rate: the paper is explicit (p. 6) that the two are not identifiable
  from the distribution of ``T``.
- ``delta_t``, ``sigma_noise`` and ``smooth_unif`` play no role: the model is
  continuous in time and has no within-trial noise.
- ``rho`` is implemented for two accumulators only, the case the paper derives.
- Sampling uses NumPy's ``default_rng``, not the C-level GSL RNG of ``cssm``,
  so ``random_state`` is comparable among LNR models but not against the
  Cython-backed models.
"""

from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators.lnr import lognormal_race

# Bounds chosen to bracket the estimates reported by the paper (Figure 6): mu
# between roughly -1.6 (true accumulators) and +1.1 (false accumulators), and
# sigma**2 between roughly 0.3 and 1.15, i.e. sigma between 0.55 and 1.07.
_MU_BOUNDS = (-3.0, 2.0)
_SIGMA_BOUNDS = (0.1, 2.0)
_T_BOUNDS = (0.0, 2.0)
_RHO_BOUNDS = (-0.95, 0.95)

# Defaults: a moderately accurate, always-terminating race (~62% choice 0).
_MU_DEFAULTS = [-1.0, -0.5, -0.5, -0.5]
_SIGMA_DEFAULTS = [0.5, 0.7, 0.7, 0.7]
_T_DEFAULT = 0.3


def _get_lnr_config(n_choices: int, correlated: bool = False) -> dict:
    """Build an LNR model config with ``n_choices`` accumulators."""
    if correlated and n_choices != 2:
        raise ValueError("Correlated LNR is only defined for two accumulators.")

    mu_params = [f"mu{i}" for i in range(n_choices)]
    sigma_params = [f"sigma{i}" for i in range(n_choices)]
    params = [*mu_params, *sigma_params]
    lower = [_MU_BOUNDS[0]] * n_choices + [_SIGMA_BOUNDS[0]] * n_choices
    upper = [_MU_BOUNDS[1]] * n_choices + [_SIGMA_BOUNDS[1]] * n_choices
    defaults = _MU_DEFAULTS[:n_choices] + _SIGMA_DEFAULTS[:n_choices]

    if correlated:
        params.append("rho")
        lower.append(_RHO_BOUNDS[0])
        upper.append(_RHO_BOUNDS[1])
        defaults.append(0.0)

    params.append("t")
    lower.append(_T_BOUNDS[0])
    upper.append(_T_BOUNDS[1])
    defaults.append(_T_DEFAULT)

    return {
        "name": f"lnr{n_choices}_corr" if correlated else f"lnr{n_choices}",
        "params": params,
        "param_bounds": [lower, upper],
        # The LNR has no evidence boundary that varies over time: the distance
        # to the boundary is folded into mu/sigma (Eq. 5).
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": len(params),
        "default_params": defaults,
        "nchoices": n_choices,
        "choices": list(range(n_choices)),
        "n_particles": n_choices,
        "tags": ["python_simulator"],
        "simulator": lognormal_race,
        "parameter_transforms": {"sampling": [], "simulation": []},
    }


def get_lnr2_config() -> dict:
    """Get configuration for the two-choice LNR (the model fit in the paper)."""
    return _get_lnr_config(n_choices=2)


def get_lnr3_config() -> dict:
    """Get configuration for the three-choice independent LNR."""
    return _get_lnr_config(n_choices=3)


def get_lnr4_config() -> dict:
    """Get configuration for the four-choice independent LNR."""
    return _get_lnr_config(n_choices=4)


def get_lnr2_corr_config() -> dict:
    """Get configuration for the two-choice LNR with correlated finishing times."""
    return _get_lnr_config(n_choices=2, correlated=True)
