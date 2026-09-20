"""Tests for the Lognormal race (LNR) models.

The LNR races ``N`` accumulators whose finishing times are Lognormal,
``T_i ~ exp(N(mu_i, sigma_i**2))``, and reports ``t + min_i T_i`` together with
the index of the winner (Heathcote & Love, 2012, Eqs. 1-5).

Correctness is checked against the paper itself -- the six quantities printed in
the Figure 2 caption and the qualitative fast-/slow-error reversal that goes
with them -- and against the paper's own likelihood (Eqs. 2-4 for the
independent race, Eqs. 9-10 for the correlated one). That likelihood is
validation material, not part of the package, so it lives in this module.

References
----------
.. [1] Heathcote, A., & Love, J. (2012). Linear deterministic accumulator
       models of simple choice. *Frontiers in Psychology, 3*, 292.
"""

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.stats import norm

from ssms import OMISSION_SENTINEL, Simulator
from ssms.basic_simulators.lnr import lognormal_race
from ssms.basic_simulators.simulator import simulator
from ssms.config import ModelConfigBuilder, get_model_registry
from ssms.config._modelconfig.lnr import _get_lnr_config

LNR_MODELS = ["lnr2", "lnr3", "lnr4", "lnr2_corr"]

# Figure 2 of the paper, (mu, sigma**2) per accumulator, shift 0.4 s. The
# caption's second entry is a *variance*: reading it as a standard deviation
# gives 78%/99% accuracy against the paper's 75%/90%.
FIG2 = {
    "speed": {"mu": (-1.2, -0.5), "var": (0.2, 0.9), "t": 0.4},
    "accuracy": {"mu": (-1.0, 0.0), "var": (0.2, 0.4), "t": 0.4},
}


# ---------------------------------------------------------------------------
# The paper's likelihood (verification only -- ssms ships simulators, not
# likelihoods, so these functions stay with the tests)
# ---------------------------------------------------------------------------


def lognormal_pdf(x, mu, sigma):
    """Eq. 3: Lognormal density."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    pos = x > 0
    z = (np.log(np.where(pos, x, 1.0)) - mu) / sigma
    out[pos] = np.exp(-0.5 * z[pos] ** 2) / (x[pos] * sigma * np.sqrt(2.0 * np.pi))
    return out


def lognormal_sf(x, mu, sigma):
    """Eq. 4: Lognormal survivor function."""
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    pos = x > 0
    out[pos] = 1.0 - norm.cdf((np.log(x[pos]) - mu) / sigma)
    return out


def defective_density(x, winner, mus, sigmas, t=0.0, rho=None):
    """Defective density of responding ``winner`` at RT ``x``.

    ``rho`` (only for two accumulators) is the correlation of the log
    finishing times; ``None``/0 gives the independent race of Eq. 2.
    """
    mus = np.asarray(mus, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    d = np.asarray(x, dtype=np.float64) - t  # decision time

    dens = lognormal_pdf(d, mus[winner], sigmas[winner])
    if rho:
        if len(mus) != 2:
            raise ValueError("rho is only defined for two accumulators.")
        other = 1 - winner
        # Eq. 9: conditional distribution of the loser given the winner's time.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_d = np.log(np.where(d > 0, d, 1.0))
        cond_mu = mus[other] + (sigmas[other] / sigmas[winner]) * rho * (
            log_d - mus[winner]
        )
        cond_sigma = np.sqrt(1.0 - rho**2) * sigmas[other]
        surv = np.ones_like(d)
        pos = d > 0
        surv[pos] = 1.0 - norm.cdf((log_d[pos] - cond_mu[pos]) / cond_sigma)
        return dens * surv

    for j in range(len(mus)):
        if j == winner:
            continue
        dens = dens * lognormal_sf(d, mus[j], sigmas[j])
    return dens


def choice_probability(winner, mus, sigmas, rho=None):
    """P(``winner`` wins the race), by quadrature over the defective density."""
    val, _ = integrate.quad(
        lambda x: defective_density(np.array([x]), winner, mus, sigmas, 0.0, rho)[0],
        0.0,
        np.inf,
        limit=400,
    )
    return val


def mean_rt(winner, mus, sigmas, t=0.0, rho=None):
    """E[RT | ``winner`` wins], including the shift ``t``."""
    p = choice_probability(winner, mus, sigmas, rho)
    num, _ = integrate.quad(
        lambda x: (
            x * defective_density(np.array([x]), winner, mus, sigmas, 0.0, rho)[0]
        ),
        0.0,
        np.inf,
        limit=400,
    )
    return t + num / p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lnr_sim():
    """Fixture providing a two-choice LNR simulator instance."""
    return Simulator(model="lnr2")


@pytest.fixture
def fig2_theta():
    """Theta dicts for the two panels of the paper's Figure 2."""

    def _make(panel):
        spec = FIG2[panel]
        return {
            "mu0": spec["mu"][0],
            "mu1": spec["mu"][1],
            "sigma0": float(np.sqrt(spec["var"][0])),
            "sigma1": float(np.sqrt(spec["var"][1])),
            "t": spec["t"],
        }

    return _make


# ---------------------------------------------------------------------------
# Registration / library integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", LNR_MODELS)
def test_models_are_registered(name):
    """Every LNR config is in the default registry and internally consistent."""
    assert get_model_registry().has_model(name)
    config = ModelConfigBuilder.from_model(name)
    ok, errors = ModelConfigBuilder.validate_config(config, strict=True)
    assert ok, errors
    assert config["n_params"] == len(config["params"])
    assert len(config["default_params"]) == len(config["params"])
    lower, upper = config["param_bounds"]
    assert all(lo < hi for lo, hi in zip(lower, upper))
    assert all(
        lo <= d <= hi for lo, d, hi in zip(lower, config["default_params"], upper)
    )


@pytest.mark.parametrize("name", LNR_MODELS)
def test_defaults_terminate(name):
    """Every trial must produce a response at the default parameters."""
    config = ModelConfigBuilder.from_model(name)
    theta = dict(zip(config["params"], config["default_params"]))
    out = simulator(model=name, theta=theta, n_samples=2000, random_state=7)
    assert np.all(out["rts"] != OMISSION_SENTINEL)
    assert np.all(out["rts"] > 0)
    assert set(np.unique(out["choices"])) <= set(config["choices"])
    # all accumulators should win at least sometimes at the defaults
    assert len(np.unique(out["choices"])) == config["nchoices"]


def test_simulator_class_path(lnr_sim, fig2_theta):
    """The high-level ``Simulator`` wrapper accepts the registered config."""
    out = lnr_sim.simulate(theta=fig2_theta("speed"), n_samples=500, random_state=3)
    assert out["rts"].shape[0] == 500
    assert np.all(out["rts"] >= FIG2["speed"]["t"])


def test_trial_wise_parameters():
    """A vector of parameters is treated as one trial per element."""
    theta = {
        "mu0": [-1.5, -0.2],
        "mu1": [-0.2, -1.5],
        "sigma0": [0.45, 0.45],
        "sigma1": [0.45, 0.45],
        "t": [0.2, 0.2],
    }
    out = simulator(model="lnr2", theta=theta, n_samples=4000, random_state=11)
    assert out["rts"].shape == (4000, 2, 1)
    # trial 0 favours accumulator 0, trial 1 favours accumulator 1
    assert out["choice_p"][0, 0] > 0.9
    assert out["choice_p"][1, 1] > 0.9


def test_omissions_are_flagged():
    """RTs past ``max_t`` are sentinel-coded in both ``rts`` and ``choices``."""
    out = simulator(
        model="lnr2",
        theta={"mu0": 1.5, "mu1": 1.5, "sigma0": 1.0, "sigma1": 1.0, "t": 0.2},
        n_samples=2000,
        max_t=2.0,
        random_state=5,
    )
    omitted = out["rts"] == OMISSION_SENTINEL
    assert omitted.any()
    assert np.all(out["choices"][omitted] == OMISSION_SENTINEL)
    assert np.all(out["rts"][~omitted] <= 2.0)


def test_deadline_variant(fig2_theta):
    """The automatic ``_deadline`` variant censors and reports omissions."""
    out = simulator(
        model="lnr2_deadline",
        theta={**fig2_theta("accuracy"), "deadline": 0.8},
        n_samples=4000,
        random_state=9,
    )
    valid = out["rts"] != OMISSION_SENTINEL
    assert np.all(out["rts"][valid] <= 0.8)
    assert out["omission_p"][0, 0] > 0.0


# ---------------------------------------------------------------------------
# Reduction: the correlated race equals the independent race at rho = 0
# ---------------------------------------------------------------------------


def test_corr_reduces_to_lnr2_at_rho_zero_identical(fig2_theta):
    """Same engine, same seed: rho = 0 gives the *same samples* as ``lnr2``.

    The correlation is applied as a Cholesky rotation of standard normal draws
    that are generated before the rotation, so at rho = 0 the rotation is the
    identity and the two models are bit-identical, not merely equal in
    distribution.
    """
    theta = fig2_theta("speed")
    a = simulator(model="lnr2", theta=theta, n_samples=5000, random_state=123)
    b = simulator(
        model="lnr2_corr", theta={**theta, "rho": 0.0}, n_samples=5000, random_state=123
    )
    np.testing.assert_array_equal(a["rts"], b["rts"])
    np.testing.assert_array_equal(a["choices"], b["choices"])


def test_nonzero_rho_changes_the_distribution(fig2_theta):
    """``rho`` is not a no-op: a correlated race has a different RT law."""
    theta = fig2_theta("speed")
    base = simulator(model="lnr2", theta=theta, n_samples=40000, random_state=17)
    corr = simulator(
        model="lnr2_corr",
        theta={**theta, "rho": 0.9},
        n_samples=40000,
        random_state=17,
    )
    # Strong positive correlation makes the race closer -> fewer errors is not
    # guaranteed, but the RT distributions must differ detectably.
    ks = stats.ks_2samp(base["rts"].ravel(), corr["rts"].ravel())
    assert ks.pvalue < 1e-6


# ---------------------------------------------------------------------------
# The simulator samples from the paper's likelihood
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("panel", ["speed", "accuracy"])
def test_marginal_rt_matches_analytic_cdf(panel, fig2_theta):
    """KS test of simulated RTs against the analytic race CDF (Eq. 2/3/4)."""
    theta = fig2_theta(panel)
    mus = (theta["mu0"], theta["mu1"])
    sigmas = (theta["sigma0"], theta["sigma1"])
    out = simulator(model="lnr2", theta=theta, n_samples=20000, random_state=2024)
    rts = out["rts"].ravel().astype(np.float64)

    # min of two independent lognormals: S(x) = S_0(x) S_1(x)
    def cdf(x):
        d = np.asarray(x) - theta["t"]
        s = np.ones_like(d)
        for mu, sigma in zip(mus, sigmas):
            s = s * np.where(
                d > 0,
                1.0 - stats.norm.cdf((np.log(np.maximum(d, 1e-12)) - mu) / sigma),
                1.0,
            )
        return 1.0 - s

    assert stats.kstest(rts, cdf).pvalue > 0.01


@pytest.mark.parametrize("panel", ["speed", "accuracy"])
def test_choice_probability_matches_analytic(panel, fig2_theta):
    """Simulated P(accumulator 0 wins) matches quadrature of Eq. 2."""
    theta = fig2_theta(panel)
    mus = (theta["mu0"], theta["mu1"])
    sigmas = (theta["sigma0"], theta["sigma1"])
    out = simulator(model="lnr2", theta=theta, n_samples=200000, random_state=99)
    p_sim = (out["choices"].ravel() == 0).mean()
    p_analytic = choice_probability(0, mus, sigmas)
    assert abs(p_sim - p_analytic) < 0.005


def test_correlated_defective_density_matches_simulation():
    """Eq. 9/10 (rho != 0) agrees with the correlated sampler.

    This exercises the one piece of the paper's mathematics that has no
    counterpart in any other library model: choice probability, conditional
    mean RT and the shape of the defective density are all compared against
    the correlated likelihood at rho = 0.6.
    """
    mus, sigmas, rho, t = (-1.0, -0.6), (0.5, 0.8), 0.6, 0.3
    out = lognormal_race(
        mu0=mus[0],
        mu1=mus[1],
        sigma0=sigmas[0],
        sigma1=sigmas[1],
        rho=rho,
        t=t,
        n_samples=200000,
        random_state=4242,
    )
    rts = out["rts"].ravel().astype(np.float64)
    choices = out["choices"].ravel()

    p0_analytic = choice_probability(0, mus, sigmas, rho=rho)
    assert abs((choices == 0).mean() - p0_analytic) < 0.005

    # defective density integrates to the choice probability, and its
    # normalised first moment to the conditional mean RT
    m0_analytic = mean_rt(0, mus, sigmas, t=t, rho=rho)
    assert abs(rts[choices == 0].mean() - m0_analytic) < 0.01

    # density itself: compare a histogram of winner-0 RTs against the bin
    # averages of the analytic defective density
    edges = np.linspace(t + 0.05, t + 1.5, 16)
    hist, _ = np.histogram(rts[choices == 0], bins=edges, density=False)
    emp = hist / (len(rts) * np.diff(edges))
    ana = np.array(
        [
            defective_density(
                np.linspace(lo, hi, 201), 0, mus, sigmas, t=t, rho=rho
            ).mean()
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )
    np.testing.assert_allclose(emp, ana, atol=0.03)


# ---------------------------------------------------------------------------
# Paper quantities (Figure 2 caption)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("panel", "acc", "mean_correct", "mean_error"),
    [("speed", 0.75, 0.70, 0.64), ("accuracy", 0.90, 0.78, 0.84)],
)
def test_figure2_quantities(panel, acc, mean_correct, mean_error, fig2_theta):
    """Reproduce the six quantities printed in the Figure 2 caption.

    The caption gives, for each emphasis condition, the accuracy and the mean
    correct/error RT predicted by the plotted parameters. The paper prints them
    to two decimals, so the tolerance is the printing precision (0.01).
    """
    theta = fig2_theta(panel)
    out = simulator(model="lnr2", theta=theta, n_samples=400000, random_state=1234)
    rts = out["rts"].ravel()
    choices = out["choices"].ravel()
    assert abs((choices == 0).mean() - acc) < 0.01
    assert abs(rts[choices == 0].mean() - mean_correct) < 0.01
    assert abs(rts[choices == 1].mean() - mean_error) < 0.01


def test_fast_errors_under_speed_slow_errors_under_accuracy(fig2_theta):
    """The qualitative claim of Figure 2: error speed reverses with emphasis.

    A larger sigma on the false accumulator (speed emphasis) makes errors fast;
    a smaller one (accuracy emphasis) makes them slow. Nothing but sigma
    changes sign here, which is the paper's point about the LNR explaining
    speed-accuracy trade-off through accumulation rather than a boundary.
    """
    res = {}
    for panel in ("speed", "accuracy"):
        out = simulator(
            model="lnr2", theta=fig2_theta(panel), n_samples=200000, random_state=55
        )
        rts, choices = out["rts"].ravel(), out["choices"].ravel()
        res[panel] = (rts[choices == 0].mean(), rts[choices == 1].mean())
    assert res["speed"][1] < res["speed"][0]  # fast errors
    assert res["accuracy"][1] > res["accuracy"][0]  # slow errors


# ---------------------------------------------------------------------------
# Structural properties of the model class
# ---------------------------------------------------------------------------


def test_no_non_responses_unlike_lba():
    """Lognormal rates are positive, so the LNR always responds (p. 5)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        theta = {
            "mu0": rng.uniform(-3, 2),
            "mu1": rng.uniform(-3, 2),
            "sigma0": rng.uniform(0.1, 2.0),
            "sigma1": rng.uniform(0.1, 2.0),
            "t": 0.0,
        }
        out = simulator(
            model="lnr2", theta=theta, n_samples=1000, max_t=1e6, random_state=1
        )
        assert np.all(out["rts"] != OMISSION_SENTINEL)


def test_single_accumulator_is_shifted_lognormal():
    """With one racer the LNR predicts shifted Lognormal simple RT (p. 5)."""
    out = lognormal_race(mu0=-0.7, sigma0=0.4, t=0.25, n_samples=50000, random_state=8)
    rts = out["rts"].ravel().astype(np.float64)
    assert (
        stats.kstest(rts - 0.25, "lognorm", args=(0.4, 0, np.exp(-0.7))).pvalue > 0.01
    )


def test_high_accuracy_limit_is_lognormal():
    """When accuracy is high the race does not distort the RT distribution."""
    out = lognormal_race(
        mu0=-1.0,
        mu1=3.0,
        sigma0=0.4,
        sigma1=0.4,
        t=0.0,
        n_samples=50000,
        random_state=6,
    )
    rts = out["rts"].ravel().astype(np.float64)
    assert (out["choices"].ravel() == 0).mean() > 0.999
    assert stats.kstest(rts, "lognorm", args=(0.4, 0, np.exp(-1.0))).pvalue > 0.01


def test_invalid_parameters_raise():
    """The documented guards reject parameters the model is not defined for.

    ``sigma`` is a standard deviation, so it must be positive; ``rho`` is a
    correlation, so it must lie inside (-1, 1); the correlated race is derived
    only for two accumulators (Eqs. 8-10); every accumulator needs a matching
    ``mu``/``sigma`` pair; and there must be at least one accumulator.
    """
    with pytest.raises(ValueError, match="sigma parameters must be strictly positive"):
        lognormal_race(mu0=-1.0, mu1=-0.5, sigma0=0.0, sigma1=0.5, n_samples=10)

    with pytest.raises(ValueError, match="rho must lie strictly between -1 and 1"):
        lognormal_race(
            mu0=-1.0, mu1=-0.5, sigma0=0.5, sigma1=0.5, rho=1.0, n_samples=10
        )

    with pytest.raises(ValueError, match="two-accumulator case"):
        lognormal_race(
            mu0=-1.0,
            mu1=-0.5,
            mu2=-0.5,
            sigma0=0.5,
            sigma1=0.5,
            sigma2=0.5,
            rho=0.5,
            n_samples=10,
        )

    with pytest.raises(ValueError, match="one mu/sigma pair is needed"):
        lognormal_race(mu0=-1.0, mu1=-0.5, sigma0=0.5, n_samples=10)

    with pytest.raises(ValueError, match="must be scalar or length n_trials"):
        lognormal_race(
            mu0=[-1.0, -0.5, 0.0],
            mu1=-0.5,
            sigma0=0.5,
            sigma1=0.5,
            n_trials=2,
            n_samples=10,
        )

    with pytest.raises(ValueError, match="At least one accumulator"):
        lognormal_race(mu0=None, sigma0=None, n_samples=10)


def test_correlated_config_is_two_choice_only():
    """The correlated race (Eqs. 8-10) is defined for two accumulators only."""
    with pytest.raises(ValueError, match="only defined for two accumulators"):
        _get_lnr_config(n_choices=3, correlated=True)


def test_lan_training_data_pipeline():
    """The config works with the library's own training-data generator."""
    from ssms.config import get_default_generator_config
    from ssms.dataset_generators.lan_mlp import TrainingDataGenerator

    config = get_default_generator_config(model="lnr2")
    config["pipeline"]["n_parameter_sets"] = 10
    config["pipeline"]["n_subruns"] = 1
    config["pipeline"]["n_cpus"] = 1
    config["simulator"]["n_samples"] = 2000
    config["training"]["n_samples_per_param"] = 100

    data = TrainingDataGenerator(config).generate_data_training(save=False)
    assert data["theta"].shape == (10, 5)
    assert data["lan_data"].shape[1] == 7
    assert np.isfinite(data["lan_labels"]).all()
