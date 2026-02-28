"""
Statistical equivalence tests: sequential (n_threads=1) vs parallel (n_threads>1) paths.

All tests are marked ``@pytest.mark.statistical`` and are skipped by default.
Run with::

    uv run pytest tests/test_parallel_equivalence.py --run-statistical -v

Design notes
------------
- Sequential path uses NumPy RNG; parallel path uses GSL Ziggurat.
  They use different RNGs so results are NOT bit-identical — the tests
  compare statistical properties (mean RT, std RT, choice proportions, KS test).
- n_samples=5000 per call gives stable enough statistics for the tolerances used
  (mean RT tol=0.05s, std RT tol=0.10s, choice prop tol=0.03, KS p>0.01).
- STD_TOL is set to 0.10 as a conservative bound. Both paths are unbiased
  (mean diff ≈ 0.004s across 10 seeds), but the per-seed sampling variability
  in RT std is ~0.023s (sd). The worst case observed across 10 seeds was
  ~0.063s (seed 42), so STD_TOL=0.10 provides ~4-sigma margin against false
  positives. The KS test provides an independent check that the full
  distributions are statistically indistinguishable.
- Models that warn and fall back to sequential (MIC2, tradeoff) are tested for:
  (a) warning is raised, (b) output is structurally valid.
- Race models have a true parallel path for n_particles <= MAX_PARTICLES and
  are given full equivalence tests.
"""

import numpy as np
import pytest
from scipy import stats

from ssms.basic_simulators.simulator import simulator
from ssms.config import model_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SAMPLES = 5000
SEED = 42
N_THREADS_PAR = 4

RT_TOL = 0.05  # Max allowed |mean_seq - mean_par| in seconds
STD_TOL = 0.10  # Max allowed |std_seq - std_par| in seconds
CHOICE_TOL = 0.03  # Max allowed |P(choice=1)_seq - P(choice=1)_par|
KS_P_MIN = 0.01  # Minimum acceptable KS test p-value


def _get_theta(model: str) -> dict:
    """Return default scalar theta dict for a model."""
    cfg = model_config[model]
    return {p: cfg["default_params"][i] for i, p in enumerate(cfg["params"])}


def _run_both(model: str) -> tuple[dict, dict]:
    """Run simulator with n_threads=1 and N_THREADS_PAR; return (seq, par)."""
    theta = _get_theta(model)
    seq = simulator(
        model=model,
        theta=theta,
        n_samples=N_SAMPLES,
        random_state=SEED,
        n_threads=1,
    )
    par = simulator(
        model=model,
        theta=theta,
        n_samples=N_SAMPLES,
        random_state=SEED,
        n_threads=N_THREADS_PAR,
    )
    return seq, par


def _assert_equivalent(seq: dict, par: dict, *, model: str = "") -> None:
    """Assert statistical equivalence between sequential and parallel outputs."""
    seq_rts = seq["rts"].ravel()
    par_rts = par["rts"].ravel()
    seq_finite = seq_rts[seq_rts > 0]
    par_finite = par_rts[par_rts > 0]

    rt_diff = abs(seq_finite.mean() - par_finite.mean())
    assert rt_diff < RT_TOL, (
        f"[{model}] Mean RT: seq={seq_finite.mean():.4f} "
        f"par={par_finite.mean():.4f} diff={rt_diff:.4f} (tol={RT_TOL})"
    )

    std_diff = abs(seq_finite.std() - par_finite.std())
    assert std_diff < STD_TOL, (
        f"[{model}] Std RT: seq={seq_finite.std():.4f} "
        f"par={par_finite.std():.4f} diff={std_diff:.4f} (tol={STD_TOL})"
    )

    seq_choices = seq["choices"].ravel()
    par_choices = par["choices"].ravel()
    unique_choices = np.unique(seq_choices[np.isfinite(seq_choices)])
    if len(unique_choices) == 2:
        choice_diff = abs((seq_choices == 1).mean() - (par_choices == 1).mean())
        assert choice_diff < CHOICE_TOL, (
            f"[{model}] Choice proportion diff={choice_diff:.4f} (tol={CHOICE_TOL})"
        )

    ks_stat, p_val = stats.ks_2samp(seq_finite, par_finite)
    assert p_val > KS_P_MIN, (
        f"[{model}] KS test: stat={ks_stat:.4f} p={p_val:.4f} (min={KS_P_MIN})"
    )


def _assert_valid_output(out: dict, *, model: str = "") -> None:
    """Assert that a simulator output dict has correct structure."""
    assert isinstance(out, dict), f"[{model}] Output is not a dict"
    assert "rts" in out, f"[{model}] 'rts' missing from output"
    assert "choices" in out, f"[{model}] 'choices' missing from output"
    assert "metadata" in out, f"[{model}] 'metadata' missing from output"
    assert np.isfinite(out["rts"]).any(), f"[{model}] No finite RTs in output"


# ---------------------------------------------------------------------------
# DDM models (ddm_models.pyx) — full parallel support
# ---------------------------------------------------------------------------


class TestDDMParallelEquivalence:
    """Equivalence tests for DDM models (ddm_flexbound backend)."""

    @pytest.mark.statistical
    @pytest.mark.parametrize("model", ["ddm", "angle", "weibull", "full_ddm"])
    def test_rt_distribution(self, model):
        """Sequential and parallel produce statistically equivalent RT distributions."""
        seq, par = _run_both(model)
        _assert_equivalent(seq, par, model=model)

    @pytest.mark.statistical
    @pytest.mark.parametrize("model", ["ddm", "angle", "weibull", "full_ddm"])
    def test_reproducibility_sequential(self, model):
        """Same seed with n_threads=1 produces identical results."""
        theta = _get_theta(model)
        r1 = simulator(
            model=model,
            theta=theta,
            n_samples=200,
            random_state=SEED,
            n_threads=1,
        )
        r2 = simulator(
            model=model,
            theta=theta,
            n_samples=200,
            random_state=SEED,
            n_threads=1,
        )
        np.testing.assert_array_equal(r1["rts"], r2["rts"])
        np.testing.assert_array_equal(r1["choices"], r2["choices"])

    @pytest.mark.statistical
    @pytest.mark.parametrize("model", ["ddm", "angle", "weibull", "full_ddm"])
    def test_reproducibility_parallel(self, model):
        """Same seed with n_threads=4 produces identical results."""
        theta = _get_theta(model)
        r1 = simulator(
            model=model,
            theta=theta,
            n_samples=200,
            random_state=SEED,
            n_threads=N_THREADS_PAR,
        )
        r2 = simulator(
            model=model,
            theta=theta,
            n_samples=200,
            random_state=SEED,
            n_threads=N_THREADS_PAR,
        )
        np.testing.assert_array_equal(r1["rts"], r2["rts"])
        np.testing.assert_array_equal(r1["choices"], r2["choices"])


# ---------------------------------------------------------------------------
# Levy models (levy_models.pyx) — full parallel support
# ---------------------------------------------------------------------------


class TestLevyParallelEquivalence:
    """Equivalence tests for Levy alpha-stable models."""

    @pytest.mark.statistical
    @pytest.mark.parametrize("model", ["levy", "levy_angle"])
    def test_rt_distribution(self, model):
        """Sequential and parallel produce statistically equivalent RT distributions."""
        seq, par = _run_both(model)
        _assert_equivalent(seq, par, model=model)

    @pytest.mark.statistical
    @pytest.mark.parametrize("model", ["levy", "levy_angle"])
    def test_reproducibility_parallel(self, model):
        """Same seed with n_threads=4 produces identical results."""
        theta = _get_theta(model)
        r1 = simulator(
            model=model,
            theta=theta,
            n_samples=200,
            random_state=SEED,
            n_threads=N_THREADS_PAR,
        )
        r2 = simulator(
            model=model,
            theta=theta,
            n_samples=200,
            random_state=SEED,
            n_threads=N_THREADS_PAR,
        )
        np.testing.assert_array_equal(r1["rts"], r2["rts"])


# ---------------------------------------------------------------------------
# Ornstein models (ornstein_models.pyx) — full parallel support
# ---------------------------------------------------------------------------


class TestOrnsteinParallelEquivalence:
    """Equivalence tests for Ornstein-Uhlenbeck models."""

    @pytest.mark.statistical
    @pytest.mark.parametrize("model", ["ornstein", "ornstein_angle"])
    def test_rt_distribution(self, model):
        """Sequential and parallel produce statistically equivalent RT distributions."""
        seq, par = _run_both(model)
        _assert_equivalent(seq, par, model=model)

    @pytest.mark.statistical
    @pytest.mark.parametrize("model", ["ornstein", "ornstein_angle"])
    def test_reproducibility_parallel(self, model):
        """Same seed with n_threads=4 produces identical results."""
        theta = _get_theta(model)
        r1 = simulator(
            model=model,
            theta=theta,
            n_samples=200,
            random_state=SEED,
            n_threads=N_THREADS_PAR,
        )
        r2 = simulator(
            model=model,
            theta=theta,
            n_samples=200,
            random_state=SEED,
            n_threads=N_THREADS_PAR,
        )
        np.testing.assert_array_equal(r1["rts"], r2["rts"])


# ---------------------------------------------------------------------------
# Parallel models (parallel_models.pyx) — full parallel support
# ---------------------------------------------------------------------------


class TestParallelModelsEquivalence:
    """Equivalence tests for Par2 models (parallel_models.pyx)."""

    @pytest.mark.statistical
    @pytest.mark.parametrize("model", ["ddm_par2", "ddm_par2_no_bias"])
    def test_rt_distribution(self, model):
        """Sequential and parallel produce statistically equivalent RT distributions."""
        seq, par = _run_both(model)
        _assert_equivalent(seq, par, model=model)

    @pytest.mark.statistical
    @pytest.mark.parametrize("model", ["ddm_par2", "ddm_par2_no_bias"])
    def test_reproducibility_parallel(self, model):
        """Same seed with n_threads=4 produces identical results."""
        theta = _get_theta(model)
        r1 = simulator(
            model=model,
            theta=theta,
            n_samples=200,
            random_state=SEED,
            n_threads=N_THREADS_PAR,
        )
        r2 = simulator(
            model=model,
            theta=theta,
            n_samples=200,
            random_state=SEED,
            n_threads=N_THREADS_PAR,
        )
        np.testing.assert_array_equal(r1["rts"], r2["rts"])


# ---------------------------------------------------------------------------
# Sequential models (sequential_models.pyx)
# ddm_flexbound_seq2 has a real parallel path; MIC2 models warn and fall back.
# ---------------------------------------------------------------------------


class TestSequentialModelsParallel:
    """Tests for sequential models with n_threads>1."""

    @pytest.mark.statistical
    @pytest.mark.parametrize("model", ["ddm_seq2", "ddm_seq2_no_bias"])
    def test_rt_distribution(self, model):
        """ddm_flexbound_seq2 has a true parallel path — verify equivalence."""
        seq, par = _run_both(model)
        _assert_equivalent(seq, par, model=model)

    @pytest.mark.statistical
    @pytest.mark.parametrize("model", ["ddm_mic2_adj", "ddm_mic2_leak"])
    def test_mic2_warns_and_produces_valid_output(self, model):
        """MIC2 models warn for n_threads>1 and fall back to sequential."""
        theta = _get_theta(model)
        with pytest.warns(UserWarning, match="n_threads"):
            out = simulator(
                model=model,
                theta=theta,
                n_samples=50,
                n_threads=N_THREADS_PAR,
            )
        _assert_valid_output(out, model=model)


# ---------------------------------------------------------------------------
# Race models (race_models.pyx) — full parallel support for n_particles <= 16
# ---------------------------------------------------------------------------


class TestRaceModelsParallel:
    """Equivalence tests for race models (race_models.pyx parallel path)."""

    @pytest.mark.statistical
    @pytest.mark.parametrize("model", ["race_no_bias_2", "race_no_bias_angle_2"])
    def test_rt_distribution(self, model):
        """Race models support parallel for n_particles <= MAX_PARTICLES (16)."""
        seq, par = _run_both(model)
        _assert_equivalent(seq, par, model=model)

    @pytest.mark.statistical
    @pytest.mark.parametrize("model", ["race_no_bias_2", "race_no_bias_angle_2"])
    def test_reproducibility_parallel(self, model):
        """Same seed with n_threads=4 produces identical results for race models."""
        theta = _get_theta(model)
        r1 = simulator(
            model=model,
            theta=theta,
            n_samples=200,
            random_state=SEED,
            n_threads=N_THREADS_PAR,
        )
        r2 = simulator(
            model=model,
            theta=theta,
            n_samples=200,
            random_state=SEED,
            n_threads=N_THREADS_PAR,
        )
        np.testing.assert_array_equal(r1["rts"], r2["rts"])


# ---------------------------------------------------------------------------
# n_threads parameter sweep
# ---------------------------------------------------------------------------


class TestNThreadsSweep:
    """Verify n_threads values 1, 2, 3, 4 all produce valid output."""

    @pytest.mark.statistical
    @pytest.mark.parametrize("n_threads", [1, 2, 3, 4])
    def test_ddm_n_threads_sweep(self, n_threads):
        """ddm model produces valid output for n_threads in {1,2,3,4}."""
        theta = _get_theta("ddm")
        out = simulator(model="ddm", theta=theta, n_samples=100, n_threads=n_threads)
        _assert_valid_output(out, model=f"ddm[n_threads={n_threads}]")

    @pytest.mark.statistical
    @pytest.mark.parametrize("n_threads", [1, 2, 3, 4])
    def test_levy_n_threads_sweep(self, n_threads):
        """levy model produces valid output for n_threads in {1,2,3,4}."""
        theta = _get_theta("levy")
        out = simulator(model="levy", theta=theta, n_samples=100, n_threads=n_threads)
        _assert_valid_output(out, model=f"levy[n_threads={n_threads}]")
