"""The racing diffusion simulator stops on a full Euler step.

The RDM (Tillman et al., 2020) is a race of independent Wiener accumulators.
Two properties follow from the model itself, independent of ``delta_t``:

* With identical parameters for every accumulator the race is exchangeable,
  so each accumulator wins with probability ``1 / n``. The Euler scheme keeps
  that symmetry, so any departure is an implementation artefact.
* Without diffusion noise an accumulator starting at 0 sits at ``v * t``; it
  cannot reach ``b`` before ``b / v``, and on the Euler grid the crossing is
  detected at the first step ``k`` with ``k * v * delta_t >= b``.

The simulator used to test each accumulator for a crossing right after moving
it and stop there: lower indices won every step in which several accumulators
crossed, and the RT omitted the step that produced the crossing.
"""

import numpy as np
import pytest
from cssm._openmp_status import check_parallel_request
from scipy.stats import chisquare

from ssms.basic_simulators.simulator import simulator

# Without OpenMP and GSL, n_threads=2 falls back to the sequential path
needs_parallel = pytest.mark.skipif(
    check_parallel_request(2, warn=False) != 2,
    reason="requires OpenMP and GSL",
)
N_THREADS = [1, pytest.param(2, marks=needs_parallel)]


@pytest.mark.parametrize("n_threads", N_THREADS)
def test_identical_accumulators_win_equally_often(n_threads):
    out = simulator(
        theta={"v0": 1.0, "v1": 1.0, "v2": 1.0, "A": 0.5, "b": 1.5, "t": 0.3},
        model="racing_diffusion_3",
        n_samples=60_000,
        delta_t=0.05,  # a coarse grid makes same-step crossings frequent
        smooth_unif=False,
        random_state=11,
        n_threads=n_threads,
    )
    choices = out["choices"].ravel()
    counts = np.array([(choices == c).sum() for c in range(3)])

    assert counts.sum() == choices.size
    assert chisquare(counts).pvalue > 1e-3, counts / counts.sum()


@pytest.mark.parametrize("n_threads", N_THREADS)
def test_noise_free_rt_includes_the_crossing_step(n_threads):
    v, b, t, delta_t = 1.0, 0.9955, 0.3, 0.01
    out = simulator(
        theta={"v0": v, "v1": 0.0, "v2": 0.0, "A": 0.0, "b": b, "t": t},
        model="racing_diffusion_3",
        n_samples=5,
        delta_t=delta_t,
        no_noise=True,
        smooth_unif=False,
        random_state=1,
        n_threads=n_threads,
    )
    decision_times = out["rts"].ravel() - t

    np.testing.assert_array_equal(out["choices"].ravel(), 0)
    # Never before the accumulator can reach b, and on the step that crosses it
    assert (decision_times >= b / v).all(), decision_times
    np.testing.assert_allclose(
        decision_times, np.ceil(b / (v * delta_t)) * delta_t, atol=1e-5
    )


@pytest.mark.parametrize("n_threads", N_THREADS)
def test_same_step_crossers_furthest_one_wins(n_threads):
    b, delta_t, t = 0.9955, 0.01, 0.3
    out = simulator(
        theta={"v0": 1.0, "v1": 1.005, "v2": 0.0, "A": 0.0, "b": b, "t": t},
        model="racing_diffusion_3",
        n_samples=5,
        delta_t=delta_t,
        no_noise=True,
        smooth_unif=False,
        random_state=1,
        n_threads=n_threads,
    )

    # Accumulators 0 and 1 both first cross on step 100, at 1.0 and 1.005
    np.testing.assert_array_equal(out["choices"].ravel(), 1)
    np.testing.assert_allclose(out["rts"].ravel() - t, 100 * delta_t, atol=1e-5)
