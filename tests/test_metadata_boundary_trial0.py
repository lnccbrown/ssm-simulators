"""Full-return metadata['boundary'] must describe trial 0 on every code path.

Regression tests for https://github.com/lnccbrown/ssm-simulators/issues/315:
the sequential kernels used to return the scratch boundary buffer, which held
the LAST trial's boundary after the trial loop, while metadata['trajectory']
(and the parallel dispatchers) described trial 0. These tests cover one model
per touched kernel family and assert the trial-0 contract on both the
sequential (n_threads=1) and parallel (n_threads>1) paths.
"""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import simulator
from ssms.config import model_config

# One model per kernel family touched by the fix, with per-row overrides that
# vary ONLY boundary parameters strongly between trial 0 and trial 1.
# model -> {param_name: (row0_value, row1_value)}
MODEL_BOUNDARY_OVERRIDES = {
    # ddm_models.pyx (ddm_flexbound)
    "angle": {"theta": (0.0, 1.2)},
    "weibull_cdf": {"alpha": (3.0, 1.0), "beta": (3.0, 1.0)},
    # sequential_models.pyx (ddm_flexbound_seq2)
    "ddm_seq2_angle_no_bias": {"theta": (0.0, 1.0)},
    # race_models.pyx (race_model)
    "race_no_bias_angle_4": {"theta": (0.0, 1.0)},
    # levy_models.pyx (levy_flexbound)
    "levy_angle": {"theta": (0.0, 1.0)},
    # ornstein_models.pyx (ornstein_uhlenbeck)
    "ornstein_angle": {"theta": (0.0, 1.0)},
    # parallel_models.pyx (ddm_flexbound_par2)
    "ddm_par2_angle_no_bias": {"theta": (0.0, 1.0)},
}

MODELS = list(MODEL_BOUNDARY_OVERRIDES.keys())


def _theta_rows(model):
    """Two parameter rows from model_config defaults, differing only in
    boundary parameters."""
    config = model_config[model]
    row0 = dict(zip(config["params"], config["default_params"]))
    row1 = dict(row0)
    for param, (value0, value1) in MODEL_BOUNDARY_OVERRIDES[model].items():
        assert param in row0, f"{model}: unknown boundary parameter {param}"
        row0[param] = value0
        row1[param] = value1
    return row0, row1


def _simulate_boundary(model, rows, n_threads=1):
    """Run the simulator on the given list of parameter rows and return
    metadata['boundary']."""
    theta = {
        key: np.array([row[key] for row in rows], dtype=np.float32) for key in rows[0]
    }
    out = simulator(
        model=model,
        theta=theta,
        n_samples=2,
        random_state=42,
        n_threads=n_threads,
    )
    return np.asarray(out["metadata"]["boundary"])


@pytest.mark.parametrize("model", MODELS)
def test_boundary_metadata_is_trial0(model):
    """Multi-trial metadata['boundary'] equals the single-trial boundary of
    the FIRST theta row -- on the sequential path."""
    row0, row1 = _theta_rows(model)

    boundary_row0 = _simulate_boundary(model, [row0])
    boundary_row1 = _simulate_boundary(model, [row1])

    # Sanity: the two rows must actually produce different boundaries,
    # otherwise the assertions below would be vacuous.
    assert not np.allclose(boundary_row0, boundary_row1), (
        f"{model}: boundary overrides did not change the boundary; "
        "test parametrization is broken"
    )

    # Trial 0 comes first: metadata documents row0, not the last row.
    boundary_multi = _simulate_boundary(model, [row0, row1])
    np.testing.assert_allclose(boundary_multi, boundary_row0, rtol=1e-6)


@pytest.mark.parametrize("model", MODELS)
def test_boundary_metadata_is_first_row_not_coincidence(model):
    """Reversing the row order flips which boundary is reported, proving the
    contract is 'first row', not an accident of ordering."""
    row0, row1 = _theta_rows(model)

    boundary_row1 = _simulate_boundary(model, [row1])
    boundary_multi_reversed = _simulate_boundary(model, [row1, row0])
    np.testing.assert_allclose(boundary_multi_reversed, boundary_row1, rtol=1e-6)


@pytest.mark.parametrize("model", MODELS)
def test_boundary_metadata_is_thread_invariant(model):
    """metadata['boundary'] is identical for n_threads=1 and n_threads=4,
    even though RTs may differ across thread counts (different RNG paths)."""
    row0, row1 = _theta_rows(model)

    boundary_seq = _simulate_boundary(model, [row0, row1], n_threads=1)
    boundary_par = _simulate_boundary(model, [row0, row1], n_threads=4)
    np.testing.assert_allclose(boundary_par, boundary_seq, rtol=1e-6)
