"""The Broadie-Glasserman-Kou boundary shift removes the Euler omission bias."""

import numpy as np
from scipy.stats import invgauss

import cssm

# Two independent accumulators with constant parameters, split into two
# artificial stages so the node logic runs. (mu, sigma, boundary0, slope, x0)
RUNNERS = [(0.55, 0.80, 1.45, -0.12, 0.10), (0.30, 0.95, 1.30, 0.06, -0.05)]
HORIZON = 2.0
NODE = 0.7
BETA = 0.5825971579390107
N_TRIALS = 100_000
DT = 2e-3


def _inputs():
    k = len(RUNNERS)
    shape = (1, k, 2)
    mu, sigma, nodes = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    intercept, slope, x0 = np.zeros(shape), np.zeros(shape), np.zeros((1, k))
    for i, (m, s, a, b, x) in enumerate(RUNNERS):
        mu[0, i], sigma[0, i], slope[0, i], x0[0, i] = m, s, b, x
        nodes[0, i] = [0.0, NODE]
        intercept[0, i] = [a, a + b * NODE]
    return dict(
        mu_array=mu,
        sigma_array=sigma,
        node_array=nodes,
        d_array=np.full((1, k), 2, dtype=np.int32),
        upper_intercept_array=intercept,
        upper_slope_array=slope,
        x0_array=x0,
    )


def _exact_omission(shift_per_sigma=0.0):
    """Product of the inverse-Gaussian survival functions at the horizon."""
    p = 1.0
    for m, s, a, b, x in RUNNERS:
        distance = a - x + shift_per_sigma * s
        relative_drift = m - b
        shape = distance**2 / s**2
        mean = distance / relative_drift
        p *= invgauss(mu=mean / shape, scale=shape).sf(HORIZON)
    return p


def _omission(boundary_correction, seed=11):
    out = cssm.race_multistage(
        **_inputs(),
        n_samples=N_TRIALS,
        delta_t=DT,
        max_t=HORIZON,
        deadline=HORIZON,
        return_option="minimal",
        random_state=seed,
        boundary_correction=boundary_correction,
    )
    return float((np.asarray(out["rts"]) == -999.0).mean())


def _standard_error(p):
    return float(np.sqrt(p * (1 - p) / N_TRIALS))


def test_corrected_kernel_matches_the_exact_omission_probability():
    exact = _exact_omission()
    assert abs(_omission(True) - exact) < 4 * _standard_error(exact)


def test_uncorrected_kernel_shows_the_discrete_monitoring_bias():
    exact = _exact_omission()
    se = _standard_error(exact)
    bias = _omission(False) - exact
    assert bias > 3 * se
    # The bias is the BGK amount: a boundary moved outward by beta*sigma*sqrt(dt).
    predicted = _exact_omission(BETA * np.sqrt(DT)) - exact
    assert abs(bias - predicted) < 4 * se


def test_correction_is_inert_without_noise():
    inputs = _inputs()
    inputs["sigma_array"][:] = 0.0
    runs = [
        cssm.race_multistage(
            **inputs,
            n_samples=2,
            delta_t=0.1,
            max_t=HORIZON,
            random_state=1,
            boundary_correction=flag,
        )
        for flag in (True, False)
    ]
    np.testing.assert_array_equal(runs[0]["rts"], runs[1]["rts"])
    np.testing.assert_array_equal(runs[0]["choices"], runs[1]["choices"])


def test_metadata_records_the_setting():
    out = cssm.race_multistage(
        **_inputs(),
        n_samples=1,
        max_t=HORIZON,
        random_state=1,
        boundary_correction=False,
    )
    assert out["metadata"]["boundary_correction"] is False
