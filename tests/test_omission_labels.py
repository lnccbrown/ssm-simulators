"""Omissions are marked in `rts`, not in `choices`.

The simulator records -999.0 as the reaction time of a trial that ran past its
deadline and leaves the latent choice alone. Masking on `choices` therefore
never excludes anything, and `omission_p` read 0.0 on every deadline model --
so an OPN trained from a deadline corpus would have learned a constant.
`nogo_p` was unaffected because it already tested `rts`.
"""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import simulator
from ssms.config.model_config_builder import ModelConfigBuilder

OMISSION_SENTINEL = -999.0
DEADLINE_MODELS = ["ddm_deadline", "angle_deadline", "gamma_drift_angle_deadline"]


def _simulate(model, deadline=0.8, n=8000, seed=0):
    cfg = ModelConfigBuilder.from_model(model)
    theta = np.array(cfg["default_params"], dtype=float)
    theta[cfg["params"].index("deadline")] = deadline
    theta[0] = 1.0  # a drift that resolves, so the deadline is what bites
    out = simulator(
        theta=np.tile(theta, (n, 1)), model=model, n_samples=1, random_state=seed
    )
    return np.asarray(out["choices"]).ravel(), np.asarray(out["rts"]).ravel()


def _labels(choices, rts, possible=(-1, 1)):
    """Call the label computation directly. It takes `self` but never uses it."""
    from ssms.dataset_generators.pipelines.simulation_pipeline import SimulationPipeline

    return SimulationPipeline._compute_auxiliary_labels(
        None,
        {
            "choices": choices,
            "rts": rts,
            "metadata": {"possible_choices": list(possible)},
        },
    )


@pytest.mark.parametrize("model", DEADLINE_MODELS)
def test_the_simulator_marks_omissions_in_rts_not_choices(model):
    """The premise. If this ever changes, the mask below must change with it."""
    choices, rts = _simulate(model)
    assert (rts == OMISSION_SENTINEL).any(), "no omissions: raise the drift or lower d"
    assert not (choices == OMISSION_SENTINEL).any(), (
        "choices now carry the sentinel too -- the omission mask needs revisiting"
    )


@pytest.mark.parametrize("model", DEADLINE_MODELS)
def test_omission_probability_is_not_identically_zero(model):
    choices, rts = _simulate(model)
    expected = float((rts == OMISSION_SENTINEL).mean())
    assert expected > 0.05, "the fixture must actually produce omissions"

    labels = _labels(choices, rts)
    assert labels["omission_p"][0, 0] == pytest.approx(expected, abs=1e-9)


def test_choice_probabilities_excluding_omissions_actually_exclude_them():
    choices, rts = _simulate("ddm_deadline")
    labels = _labels(choices, rts)
    kept = choices[rts != OMISSION_SENTINEL]
    for i, c in enumerate([-1, 1]):
        assert labels["choice_p_no_omission"][0, i] == pytest.approx(
            float((kept == c).mean()), abs=1e-9
        )
    # and they must differ from the all-trials version, or the test proves nothing
    assert not np.allclose(
        labels["choice_p_no_omission"][0], labels["choice_p"][0], atol=1e-6
    )


def test_nogo_decomposes_into_wrong_choice_plus_omission():
    """The identity the derived-network plan relies on, checked on real output."""
    choices, rts = _simulate("gamma_drift_angle_deadline")
    labels = _labels(choices, rts)
    omitted = rts == OMISSION_SENTINEL
    p_min = float(((choices != 1) & ~omitted).mean())
    p_omit = float(omitted.mean())
    assert labels["nogo_p"][0, 0] == pytest.approx(p_min + p_omit, abs=1e-9)
