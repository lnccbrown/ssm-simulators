"""Tests for the ddm_normal_st model configuration.

ddm_normal_st draws non-decision time from an untruncated Normal kernel, so `st`
is a standard deviation rather than the uniform half-width `ddm_st` uses. These
tests pin the two consequences: the dispersion semantics of `st`, and the fact
that the kernel has no floor, which makes RT positivity a matter of measured
exposure rather than a guarantee.
"""

import numpy as np
import pytest
import scipy.stats as sps

from ssms.basic_simulators.simulator import simulator
from ssms.config import KDE_NO_DISPLACE_T, model_config
from ssms.config.generator_config.data_generator_config import (
    get_default_generator_config,
)
from ssms.dataset_generators.lan_mlp import TrainingDataGenerator

# Every non-positive-RT rate below is a Monte Carlo estimate: the rate is a
# property of an unbounded kernel, and `random_state` does not pin the ndt
# draws (t_dist samples from scipy's global RNG). Tolerances are set from a
# 3-seed x 200_000-sample measurement, whose seed-to-seed spread at the box
# corner is +-0.06 percentage points.
N_SAMPLES = 100_000

# The lower `t` bound with the widest `st`, the worst corner of the box.
BOX_CORNER = {"v": 0.0, "a": 0.3, "z": 0.5, "t": 0.25, "st": 0.25}

# `st` at which the kernel's 3-sigma floor sits exactly on rt = 0 at t = 0.25.
THREE_SIGMA_ST = BOX_CORNER["t"] / 3.0


def _frac_nonpositive(theta, model="ddm_normal_st", n_samples=N_SAMPLES):
    """Fraction of simulated RTs that a KDE cannot take the log of."""
    rts = simulator(model=model, theta=theta, n_samples=n_samples)["rts"]
    return float(np.mean(rts <= 0.0))


def test_ddm_normal_st_is_registered_over_ddm_st_s_box():
    """ddm_normal_st exposes ddm_st's parameters over ddm_st's bounds."""
    config = model_config["ddm_normal_st"]
    assert config["name"] == "ddm_normal_st"
    assert config["params"] == model_config["ddm_st"]["params"]
    np.testing.assert_array_equal(
        np.asarray(config["param_bounds"], dtype=float),
        np.asarray(model_config["ddm_st"]["param_bounds"], dtype=float),
    )


def test_ddm_normal_st_simulates():
    """Smoke test at a parameter set the 3-sigma floor keeps clear of zero."""
    result = simulator(
        model="ddm_normal_st",
        theta={"v": 1.0, "a": 1.5, "z": 0.5, "t": 1.0, "st": 0.1},
        n_samples=100,
    )

    assert result["rts"].shape == (100, 1)
    assert result["choices"].shape == (100, 1)
    assert set(np.unique(result["choices"])).issubset({-1, 1})
    assert np.all(result["rts"] > 0)


def test_st_is_a_standard_deviation_not_a_half_width():
    """`st` is the ndt kernel's SD; ddm_st's `st` is a half-width (SD st/sqrt(3))."""
    st = 0.15
    normal = sps.norm(
        **model_config["ddm_normal_st"]["simulator_param_mappings"]["t_dist"](
            st
        ).keywords
    )
    uniform = sps.uniform(
        **model_config["ddm_st"]["simulator_param_mappings"]["t_dist"](st).keywords
    )

    assert normal.std() == pytest.approx(st)
    assert uniform.std() == pytest.approx(st / np.sqrt(3.0))

    # The half-width kernel floors ndt at t - st; the Normal one has no floor,
    # which is the whole of the difference in downstream RT handling.
    assert uniform.support() == pytest.approx((-st, st))
    assert normal.support()[0] == -np.inf


def test_bounded_sibling_never_emits_a_non_positive_rt():
    """ddm_st's floor is t - st >= 0 over its whole box, and rt adds a positive dt."""
    rts = simulator(model="ddm_st", theta=BOX_CORNER, n_samples=N_SAMPLES)["rts"]
    assert np.all(rts > 0)


def test_non_positive_rts_are_a_declared_exposure_at_the_box_corner():
    """The unbounded kernel puts a large minority of RTs at or below zero here.

    Consumers that cannot represent a non-positive RT must filter, not assume.
    Measured 8.92% (8.88-9.00 over 3 seeds x 200_000 samples).
    """
    assert 0.05 < _frac_nonpositive(BOX_CORNER) < 0.15


def test_non_positive_rt_exposure_falls_but_never_reaches_zero():
    """At st = t/3 the 3-sigma floor sits on zero, and exposure is ~1e-4, not 0.

    This is why `3 * st <= t` is a rate bound and not a positivity guarantee.
    Measured 0.0128% at st = t/3 and 0 in 600_000 draws at st = 0.05.
    """
    three_sigma = _frac_nonpositive(
        dict(BOX_CORNER, st=THREE_SIGMA_ST), n_samples=200_000
    )
    assert 0.0 < three_sigma < 1e-3

    assert _frac_nonpositive(dict(BOX_CORNER, st=0.05)) < 1e-4


def test_generator_refuses_displace_t_for_ddm_normal_st():
    """An unbounded ndt kernel has no support edge to displace by, as for ddm_st."""
    assert "ddm_normal_st" in KDE_NO_DISPLACE_T

    generator_config = get_default_generator_config("lan")
    generator_config["model"] = "ddm_normal_st"
    generator_config["pipeline"]["n_cpus"] = 1
    generator_config["estimator"]["displace_t"] = True

    with pytest.warns(UserWarning, match="displace_t is True"):
        generator = TrainingDataGenerator(
            config=generator_config, model_config=model_config["ddm_normal_st"]
        )

    assert generator.generator_config["estimator"]["displace_t"] is False
