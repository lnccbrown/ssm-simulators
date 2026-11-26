"""The sampler reproduces the paper's closed-form distributions (Eqs. 1-3)."""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import simulator
from ssms.basic_simulators.trdm import first_passage_times
from ssms.support_utils.trdm_analytic import Wald, race_probabilities, response_density


@pytest.mark.parametrize(
    "v,s,d",
    [
        (1.0, 1.0, 1.0),
        (4.2, 1.0, 1.2),
        (0.0, 1.2, 1.0),
        (-0.5, 1.0, 1.0),
        (1.06, 0.14, 1.0),
        (0.05, 1.0, 2.0),
    ],
)
def test_first_passage_cdf_matches_eq3(v, s, d):
    rng = np.random.default_rng(0)
    n = 200_000
    x = first_passage_times(
        np.array(v),
        np.array(s),
        np.array(d),
        rng.standard_normal(n),
        rng.random(n),
        rng.random(n),
    )
    W = Wald(v, s, d, 0.0)
    # defective mass for negative drift
    assert np.mean(np.isfinite(x)) == pytest.approx(W.total_mass(), abs=4e-3)
    fin = x[np.isfinite(x)]
    grid = np.quantile(fin, [0.05, 0.25, 0.5, 0.75, 0.95])
    emp = np.array([np.mean(x <= g) for g in grid])
    np.testing.assert_allclose(emp, W.cdf(grid), atol=4e-3)


def test_degenerate_cases():
    z = np.array([0.3])
    u = np.array([0.4])
    h = np.array([0.2])
    # no noise: deterministic d/v
    assert first_passage_times(np.array(2.0), np.array(0.0), np.array(1.0), z, u, h)[
        0
    ] == pytest.approx(0.5)
    # no noise, no drift: never
    assert np.isinf(
        first_passage_times(np.array(0.0), np.array(0.0), np.array(1.0), z, u, h)[0]
    )
    # no noise, negative drift: never
    assert np.isinf(
        first_passage_times(np.array(-1.0), np.array(0.0), np.array(1.0), z, u, h)[0]
    )
    # starting on threshold: immediately
    assert (
        first_passage_times(np.array(1.0), np.array(1.0), np.array(0.0), z, u, h)[0]
        == 0.0
    )


@pytest.mark.parametrize(
    "theta",
    [
        {
            "v0": 2.0,
            "v1": 0.5,
            "b0": 1.0,
            "b1": 1.0,
            "t": 0.3,
            "vT": 1.0,
            "sT": 0.5,
            "tT": 0.05,
        },
        # implicit-deadlines easy condition in library units (negative error drift)
        {
            "v0": (0.83 + 4.22 / 2) / 0.49,
            "v1": (0.83 - 4.22 / 2) / 0.49,
            "b0": 0.995 / 0.49,
            "b1": 1.005 / 0.49,
            "t": 0.09,
            "vT": 1.05,
            "sT": 0.22,
            "tT": 0.05,
        },
        # timer variable and early
        {
            "v0": 1.0,
            "v1": 0.8,
            "b0": 1.4,
            "b1": 1.2,
            "t": 0.4,
            "vT": 0.1,
            "sT": 1.2,
            "tT": 0.0,
        },
    ],
)
def test_trdm_2_matches_eq1(theta):
    n = 200_000
    out = simulator(theta=theta, model="trdm_2", n_samples=n, random_state=11)
    rt = np.asarray(out["rts"]).ravel()
    ch = np.asarray(out["choices"]).ravel()
    ev = [
        Wald(theta["v0"], 1.0, theta["b0"], theta["t"]),
        Wald(theta["v1"], 1.0, theta["b1"], theta["t"]),
    ]
    tm = Wald(theta["vT"], theta["sT"], 1.0, theta["tT"])
    r = race_probabilities(ev, tm)
    p0 = r["evidence"][0] + r["timer"] / 2
    assert np.mean(ch == 0) == pytest.approx(
        p0, abs=4 * np.sqrt(p0 * (1 - p0) / n) + 1e-4
    )
    # defective response-0 CDF at a few RTs vs integrated Eq. 1
    from scipy import integrate

    for q in np.quantile(rt, [0.1, 0.5, 0.9]):
        exact = integrate.quad(
            lambda x: response_density(ev, tm, 0, np.array([x]))[0],
            0.0,
            q,
            limit=400,
            points=[theta["t"], theta["tT"]],
        )[0]
        assert np.mean((ch == 0) & (rt <= q)) == pytest.approx(exact, abs=5e-3)


def test_rt_can_precede_evidence_onset():
    """The timer can finish before tau_E (paper p. 11); support starts at min(t, tT)."""
    th = {
        "v0": 1.0,
        "v1": 1.0,
        "b0": 1.0,
        "b1": 1.0,
        "t": 0.5,
        "vT": 0.1,
        "sT": 1.21,
        "tT": 0.0,
    }
    out = simulator(theta=th, model="trdm_2", n_samples=20000, random_state=2)
    rt = np.asarray(out["rts"]).ravel()
    assert np.mean(rt < 0.5) > 0.01 and rt.min() >= 0.0
