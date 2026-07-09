"""CE-1: the ported efpt aDDM simulator + aligned parameter contract.

The engine (inline xoshiro256++ PRNG + stage-indexed Euler kernel) is vendored
verbatim from efficient-fpt, so ``cssm.addm`` and HSSM's JAX aDDM likelihood share
one engine. These tests pin:

* **bit-for-bit parity vs efpt** on fixed per-trial seeds (skipped if efficient_fpt
  — a dev-only oracle, not on PyPI — is unavailable),
* **n_threads-independent determinism** (per-trial seeds),
* the **aligned param contract** ``[eta, kappa, a, b, x0, t]`` (z->x0 absolute,
  s->sigma) and the registered ``addm_collapse`` boundary,
* **x0 as an absolute start**, the **self-sampled** (Mode-1) path, and the
  **omission** sentinel.
"""

import numpy as np
import pytest

import cssm.addm_models as am
from ssms.basic_simulators import boundary_functions as bf
from ssms.config import get_boundary_registry
from ssms.config._modelconfig.addm import get_addm_config

try:
    from efficient_fpt.addm_helpers import _build_addm_mu_array_data as _efpt_mu
    from efficient_fpt.cython.simulator import _simulate_addm_fpt as _efpt_sim

    _HAS_EFPT = True
except Exception:  # pragma: no cover - oracle is optional
    _HAS_EFPT = False

needs_efpt = pytest.mark.skipif(
    not _HAS_EFPT, reason="efficient_fpt oracle not installed"
)

_ETA, _KAPPA, _SIGMA, _A, _B, _X0 = 0.3, 1.0, 1.0, 1.5, 0.25, 0.0


def _fixed_covariates(n=12, max_d=5, seed=0):
    """Deterministic observed fixations for a batch of aDDM trials."""
    rng = np.random.default_rng(seed)
    r1 = rng.integers(1, 6, n).astype(np.float64)
    r2 = rng.integers(1, 6, n).astype(np.float64)
    flag = rng.integers(0, 2, n).astype(np.int64)
    d = rng.integers(2, max_d + 1, n).astype(np.int32)
    sacc = np.zeros((n, max_d), dtype=np.float64)
    for i in range(n):
        sacc[i, 1 : d[i]] = np.sort(rng.uniform(0.1, 1.5, d[i] - 1))
    return dict(n=n, max_d=max_d, r1=r1, r2=r2, flag=flag, d=d, sacc=sacc)


def _efpt_boundary_arrays(sacc, a, b, sigma):
    """Reproduce efpt's per-stage array expansion (ub=a-b*sacc, slopes -b/+b)."""
    n, max_d = sacc.shape
    sig = np.ascontiguousarray(np.full((n, max_d), sigma))
    ub = np.ascontiguousarray(a - b * sacc)
    lb = np.ascontiguousarray(-(a - b * sacc))
    b1 = np.ascontiguousarray(np.full((n, max_d), -b))
    b2 = np.ascontiguousarray(np.full((n, max_d), b))
    return sig, ub, lb, b1, b2


# --------------------------------------------------------------------------- #
# Parity vs efpt (bit-for-bit; same seeds -> same trajectory)
# --------------------------------------------------------------------------- #
@needs_efpt
def test_addm_engine_parity_vs_efpt_fixed_seed():
    fx = _fixed_covariates()
    n, max_d = fx["n"], fx["max_d"]
    x0d = np.full(n, _X0)
    seeds = np.arange(1, n + 1, dtype=np.uint64)
    dt, T = 0.001, 10.0

    mu = np.ascontiguousarray(
        _efpt_mu(_ETA, _KAPPA, fx["r1"], fx["r2"], fx["flag"], fx["d"], max_d)
    )
    rt_e, ch_e, _ = _efpt_sim(
        mu,
        np.ascontiguousarray(fx["sacc"]),
        fx["d"],
        _SIGMA,
        _A,
        _B,
        x0d,
        dt,
        T,
        seeds,
        1,
    )

    mu2 = am._build_addm_mu_array_data(
        _ETA, _KAPPA, fx["r1"], fx["r2"], fx["flag"], fx["d"], max_d
    )
    assert np.array_equal(mu, mu2), "ported mu construction diverged from efpt"
    sig, ub, lb, b1, b2 = _efpt_boundary_arrays(fx["sacc"], _A, _B, _SIGMA)
    rt_s, ch_s, _ = am._simulate_heterog_multistage(
        np.ascontiguousarray(mu2),
        sig,
        np.ascontiguousarray(fx["sacc"]),
        fx["d"],
        ub,
        b1,
        lb,
        b2,
        np.ascontiguousarray(x0d),
        dt,
        T,
        seeds,
        1,
    )

    # Same engine + same seeds => bit-identical trajectories.
    assert np.array_equal(np.asarray(ch_e), np.asarray(ch_s))
    np.testing.assert_array_equal(np.asarray(rt_e), np.asarray(rt_s))


@needs_efpt
def test_addm_public_mode2_matches_efpt():
    """cssm.addm (Mode 2, n_samples=1) reproduces efpt on its derived seeds."""
    fx = _fixed_covariates(seed=3)
    n, max_d = fx["n"], fx["max_d"]
    random_state = 7
    dt, T = 0.001, 10.0

    # cssm.addm derives per-row seeds as default_rng(random_state).integers(...).
    seeds = np.random.default_rng(random_state).integers(
        0, 2**64, size=n, dtype=np.uint64
    )
    mu = np.ascontiguousarray(
        _efpt_mu(_ETA, _KAPPA, fx["r1"], fx["r2"], fx["flag"], fx["d"], max_d)
    )
    rt_e, ch_e, _ = _efpt_sim(
        mu,
        np.ascontiguousarray(fx["sacc"]),
        fx["d"],
        _SIGMA,
        _A,
        _B,
        np.full(n, _X0),
        dt,
        T,
        seeds,
        1,
    )

    col = lambda v: np.full(n, v, dtype=np.float64)  # noqa: E731
    out = am.addm(
        col(_ETA),
        col(_KAPPA),
        col(_A),
        col(_B),
        col(_X0),
        col(0.0),
        col(999.0),
        col(_SIGMA),
        sigma=col(_SIGMA),
        r1=fx["r1"],
        r2=fx["r2"],
        flag=fx["flag"],
        sacc_array=fx["sacc"],
        d=fx["d"],
        delta_t=dt,
        max_t=T,
        n_samples=1,
        n_trials=n,
        random_state=random_state,
        n_threads=1,
    )
    rt_s = np.asarray(out["rts"]).reshape(-1)
    ch_s = np.asarray(out["choices"]).reshape(-1)

    term = rt_e >= 0.0  # compare terminated trials (efpt -1 vs cssm -999 sentinels)
    np.testing.assert_allclose(rt_s[term], rt_e[term], atol=1e-5, rtol=0)
    assert np.array_equal(ch_s[term], np.asarray(ch_e)[term])


# --------------------------------------------------------------------------- #
# Determinism across n_threads (per-trial xoshiro seeds)
# --------------------------------------------------------------------------- #
def test_addm_deterministic_across_n_threads():
    fx = _fixed_covariates(n=40, seed=1)
    n = fx["n"]
    col = lambda v: np.full(n, v, dtype=np.float64)  # noqa: E731
    kw = dict(
        r1=fx["r1"],
        r2=fx["r2"],
        flag=fx["flag"],
        sacc_array=fx["sacc"],
        d=fx["d"],
        n_samples=1,
        n_trials=n,
        random_state=123,
        max_t=10.0,
    )
    a1 = am.addm(
        col(_ETA),
        col(_KAPPA),
        col(_A),
        col(_B),
        col(_X0),
        col(0.1),
        col(999.0),
        col(_SIGMA),
        n_threads=1,
        **kw,
    )
    a4 = am.addm(
        col(_ETA),
        col(_KAPPA),
        col(_A),
        col(_B),
        col(_X0),
        col(0.1),
        col(999.0),
        col(_SIGMA),
        n_threads=4,
        **kw,
    )
    np.testing.assert_array_equal(np.asarray(a1["rts"]), np.asarray(a4["rts"]))
    np.testing.assert_array_equal(np.asarray(a1["choices"]), np.asarray(a4["choices"]))


# --------------------------------------------------------------------------- #
# Aligned parameter contract + boundary registration
# --------------------------------------------------------------------------- #
def test_addm_collapse_boundary_registered():
    reg = get_boundary_registry()
    assert reg.is_registered("addm_collapse")
    assert reg.get("addm_collapse")["params"] == ["a", "b"]
    assert bf.addm_collapse(1.0, 2.0, 0.5) == 1.5  # a - b*t
    assert float(bf.addm_collapse(0.0, 2.0, 0.5)) == 2.0


def test_addm_param_contract_renamed():
    cfg = get_addm_config()
    assert cfg["params"] == ["eta", "kappa", "a", "b", "x0", "t"]
    assert "z" not in cfg["params"] and "s" not in cfg["params"]
    assert cfg["boundary_name"] == "addm_collapse"

    from ssms.basic_simulators.simulator import simulator

    out = simulator(
        model="addm",
        theta=np.array([[0.3, 1.0, 1.5, 0.25, 0.0, 0.1]]),  # eta,kappa,a,b,x0,t
        n_samples=50,
        random_state=1,
    )
    assert out["rts"].shape[0] == 50
    assert set(np.unique(out["choices"]).tolist()) <= {-1, 1}


def test_addm_high_level_apis_forward_extra_fields():
    """Both simulator() and the Simulator class do Mode-2 via extra_fields.

    Robust check: with the SAME seed but DIFFERENT observed fixations, the per-trial
    output must differ — the fixations are actually forwarded and used. If
    extra_fields were dropped, both calls would self-sample identical fixations from
    the same seed and produce identical output.
    """
    from ssms.basic_simulators import Simulator
    from ssms.basic_simulators.simulator import simulator

    fx_a = _fixed_covariates(n=300, seed=4)
    fx_b = _fixed_covariates(n=300, seed=9)  # different gaze pattern, same shapes
    n = fx_a["n"]

    def ef(fx):
        return {
            "r1": fx["r1"],
            "r2": fx["r2"],
            "flag": fx["flag"],
            "sacc_array": fx["sacc"],
            "d": fx["d"],
            "sigma": np.ones(n),
        }

    theta = np.tile([_ETA, _KAPPA, _A, _B, _X0, 0.0], (n, 1))
    for run in (
        lambda **kw: simulator(
            model="addm", theta=theta, n_samples=1, random_state=5, **kw
        ),
        lambda **kw: Simulator("addm").simulate(
            theta=theta, n_samples=1, random_state=5, **kw
        ),
    ):
        rt_a = np.asarray(run(extra_fields=ef(fx_a))["rts"]).reshape(-1)
        rt_b = np.asarray(run(extra_fields=ef(fx_b))["rts"]).reshape(-1)
        assert not np.array_equal(rt_a, rt_b), "fixations were not forwarded/used"


def test_addm_x0_absolute_start():
    """x0 is the ABSOLUTE start: with no drift/noise, x_final == x0 exactly."""
    n, max_d = 4, 3
    sacc = np.zeros((n, max_d))
    d = np.full(n, 1, dtype=np.int32)
    mu = np.zeros((n, max_d))
    sig = np.zeros((n, max_d))
    a = 10.0  # bounds never reached
    ub = np.full((n, max_d), a)
    lb = np.full((n, max_d), -a)
    b1 = np.zeros((n, max_d))
    b2 = np.zeros((n, max_d))
    x0 = np.array([0.5, -0.3, 0.9, 0.0])
    _, _, xf = am._simulate_heterog_multistage(
        np.ascontiguousarray(mu),
        np.ascontiguousarray(sig),
        np.ascontiguousarray(sacc),
        d,
        np.ascontiguousarray(ub),
        b1,
        np.ascontiguousarray(lb),
        b2,
        np.ascontiguousarray(x0),
        0.001,
        1.0,
        np.arange(1, n + 1, dtype=np.uint64),
        1,
    )
    np.testing.assert_array_equal(
        np.asarray(xf), x0
    )  # no movement -> start is absolute


def test_addm_self_sample_mode_still_runs():
    """Mode 1 (no covariates): self-samples fixations, reproducible per seed."""
    n = 5
    col = lambda v: np.full(n, v, dtype=np.float64)  # noqa: E731
    args = (
        col(_ETA),
        col(_KAPPA),
        col(_A),
        col(_B),
        col(_X0),
        col(0.1),
        col(999.0),
        col(_SIGMA),
    )
    kw = dict(n_samples=100, n_trials=n, random_state=42, max_t=10.0)
    a = am.addm(*args, **kw)
    b = am.addm(*args, **kw)
    rts = np.asarray(a["rts"])
    assert rts.shape == (100, n, 1)
    assert set(np.unique(a["choices"]).tolist()) <= {-1, 0, 1}
    valid = rts[rts != -999.0]
    assert valid.size > 0 and np.all(valid > 0)
    np.testing.assert_array_equal(rts, np.asarray(b["rts"]))  # reproducible


def test_addm_omission_sentinel():
    """Trials that don't terminate by max_t get rt == -999.0 and choice == 0."""
    n = 8
    fx = _fixed_covariates(n=n, seed=5)
    col = lambda v: np.full(n, v, dtype=np.float64)  # noqa: E731
    # Very short horizon + wide bounds => most trials do not cross.
    out = am.addm(
        col(0.0),
        col(0.0),
        col(5.0),
        col(0.0),
        col(0.0),
        col(0.0),
        col(999.0),
        col(0.5),
        r1=fx["r1"],
        r2=fx["r2"],
        flag=fx["flag"],
        sacc_array=fx["sacc"],
        d=fx["d"],
        n_samples=1,
        n_trials=n,
        random_state=0,
        delta_t=0.001,
        max_t=0.02,
    )
    rts = np.asarray(out["rts"]).reshape(-1)
    ch = np.asarray(out["choices"]).reshape(-1)
    omitted = rts == -999.0
    assert omitted.any(), "expected some omissions with a tiny horizon"
    assert np.all(ch[omitted] == 0)


def test_addm_cartoon_metadata_boundary_trajectory_z():
    """Model-cartoon contract: full metadata carries a collapsing boundary array,
    a recorded trajectory (row 0), and a relative ``z`` start marker (``x0=0``
    maps to ``0.5``). Holds for both the deterministic no_noise drift path (used
    by the cartoon's drift line) and a noisy sim (its example sample paths)."""
    from ssms.basic_simulators.simulator import simulator

    theta = np.array([[0.4, 1.0, 1.5, 0.2, 0.0, 0.0]])  # eta,kappa,a,b,x0,t; x0=0
    for no_noise in (True, False):
        out = simulator(
            model="addm",
            theta=theta,
            n_samples=1,
            random_state=7,
            no_noise=no_noise,
            delta_t=0.01,
            max_t=10.0,
        )
        md = out["metadata"]
        assert {"boundary", "trajectory", "z"} <= set(md), md.keys()

        b = np.asarray(md["boundary"]).reshape(-1)
        assert b.ndim == 1 and b[0] > 0.0 and b[0] >= b[-1]  # +(a - b*t) collapses

        traj = np.asarray(md["trajectory"]).reshape(-1)
        recorded = traj[traj > -999.0]
        assert recorded.size > 1  # a path was actually recorded (not all -999)
        assert abs(float(recorded[0])) < 1e-4  # traj[0] == x0 == 0 (aligned start)

        z = np.asarray(md["z"]).reshape(-1)
        assert (
            0.0 <= float(z[0]) <= 1.0 and abs(float(z[0]) - 0.5) < 1e-4
        )  # x0=0 -> mid


# --------------------------------------------------------------------------- #
# Pluggable fixation continuation (PPC) + pluggable Mode-1 distribution
# --------------------------------------------------------------------------- #
_GAMMA_CONT = {"dist": "gamma", "dist_params": {"a": 2.0, "scale": 0.3}}


def _run_mode2(n=6, seed=2, n_samples=200, random_state=1, **overrides):
    """Mode-2 aDDM on fixed observed covariates; returns the addm() output dict."""
    fx = _fixed_covariates(n=n, seed=seed)
    col = lambda v: np.full(n, v, dtype=np.float64)  # noqa: E731
    return am.addm(
        col(_ETA),
        col(_KAPPA),
        col(_A),
        col(_B),
        col(_X0),
        col(0.0),
        col(999.0),
        col(_SIGMA),
        sigma=col(_SIGMA),
        r1=fx["r1"],
        r2=fx["r2"],
        flag=fx["flag"],
        sacc_array=fx["sacc"],
        d=fx["d"],
        n_samples=n_samples,
        n_trials=n,
        random_state=random_state,
        max_t=10.0,
        **overrides,
    )


def test_addm_continuation_default_is_prolong_and_deterministic():
    """Default == explicit prolong_last_fixation, and reproducible. The isolated cont_rng means
    the default never perturbs the primary rng that seeds the trajectories (byte-identical path)."""
    a = _run_mode2()
    b = _run_mode2(continuation_mode="prolong_last_fixation")
    c = _run_mode2()
    np.testing.assert_array_equal(np.asarray(a["rts"]), np.asarray(b["rts"]))
    np.testing.assert_array_equal(np.asarray(a["choices"]), np.asarray(b["choices"]))
    np.testing.assert_array_equal(
        np.asarray(a["rts"]), np.asarray(c["rts"])
    )  # reproducible


def test_addm_sample_continuation_reaches_simulator():
    """sample_continuation draws the tail from a distribution -> different output than prolong."""
    prolong = np.asarray(_run_mode2()["rts"]).reshape(-1)
    cont = np.asarray(
        _run_mode2(
            continuation_mode="sample_continuation", continuation_params=_GAMMA_CONT
        )["rts"]
    ).reshape(-1)
    assert not np.array_equal(prolong, cont)


def test_addm_resample_all_fixations_reaches_simulator():
    """resample_all_fixations self-samples the whole schedule -> differs from prolong; valid output."""
    prolong = np.asarray(_run_mode2()["rts"]).reshape(-1)
    out = _run_mode2(
        continuation_mode="resample_all_fixations", continuation_params=_GAMMA_CONT
    )
    r = np.asarray(out["rts"]).reshape(-1)
    assert not np.array_equal(prolong, r)
    assert (r[r != -999.0] > 0).all()  # non-omission rts are positive


def test_addm_mode1_accepts_nongamma_distribution():
    """Mode-1 self-sampling now accepts any positive scipy dist (not just gamma)."""
    n = 5
    col = lambda v: np.full(n, v, dtype=np.float64)  # noqa: E731
    out = am.addm(
        col(_ETA),
        col(_KAPPA),
        col(_A),
        col(_B),
        col(_X0),
        col(0.1),
        col(999.0),
        col(_SIGMA),
        n_samples=50,
        n_trials=n,
        random_state=0,
        max_t=10.0,
        fixation_dist="lognormal",
        fixation_params={"s": 0.5, "scale": 0.4},
    )
    assert np.asarray(out["rts"]).shape == (50, n, 1)
    assert set(np.unique(out["choices"]).tolist()) <= {-1, 0, 1}


def test_addm_continuation_errors():
    """Clear errors for unknown mode, unknown distribution, and prolong + non-None params."""
    with pytest.raises(ValueError):
        _run_mode2(continuation_mode="nope")
    with pytest.raises(ValueError):
        _run_mode2(
            continuation_mode="sample_continuation",
            continuation_params={"dist": "not_a_dist"},
        )
    with pytest.raises(ValueError):
        _run_mode2(
            continuation_mode="prolong_last_fixation", continuation_params=_GAMMA_CONT
        )


def test_fixation_continuation_module_strategies():
    """Unit-test the strategy functions directly (pure Python; no simulator)."""
    from ssms.basic_simulators import fixation_continuation as fc

    rng = np.random.default_rng(0)
    node = np.array([[0.0, 0.3, 0.7], [0.0, 0.5, 0.0]])
    d = np.array([3, 2], dtype=np.int32)
    flag = np.array([0, 1], dtype=np.int64)
    p = {"dist": "gamma", "dist_params": {"a": 2.0, "scale": 0.3}}

    _, d0, f0, _ = fc.prolong_last_fixation(node, d, flag, 5.0, None, rng)
    assert np.array_equal(d0, d) and np.array_equal(f0, flag)

    n1, _, f1, m1 = fc.sample_continuation(node, d, flag, 5.0, p, rng)
    assert m1 > node.shape[1] and np.array_equal(f1, flag)
    assert (
        n1[0, d[0]] > node[0, d[0] - 1]
    )  # first continuation onset > last observed onset

    n2, _, _, _ = fc.resample_all_fixations(node, d, flag, 5.0, p, rng)
    assert bool((n2[:, 0] == 0.0).all())  # fresh schedule anchored at 0.0


def test_fixation_continuation_resolvers_and_require_dist():
    """Resolver seam (callable passthrough / bad type) + the _require_dist guard."""
    from ssms.basic_simulators import fixation_continuation as fc

    rng = np.random.default_rng(0)
    node = np.array([[0.0, 0.3, 0.7]])
    d = np.array([3], dtype=np.int32)
    flag = np.array([0], dtype=np.int64)

    # sample_continuation / resample_all_fixations require continuation_params w/ a 'dist'.
    with pytest.raises(ValueError):
        fc.sample_continuation(node, d, flag, 5.0, None, rng)
    with pytest.raises(ValueError):
        fc.resample_all_fixations(node, d, flag, 5.0, {"dist_params": {}}, rng)

    # resolve_continuation_mode: a callable is returned as-is (pluggable-strategy seam) ...
    assert (
        fc.resolve_continuation_mode(fc.prolong_last_fixation)
        is fc.prolong_last_fixation
    )
    # ... and a non-str / non-callable is rejected loudly.
    with pytest.raises(TypeError):
        fc.resolve_continuation_mode(123)
