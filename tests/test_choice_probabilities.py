"""Per-trial choice / omission / go-nogo probabilities computed by ``simulator()``.

``_choice_probabilities`` replaced a per-trial Python loop (issue #79). The loop
is kept below as the reference implementation, and the outputs must match it
exactly: they feed HSSM, so even a last-ulp difference would be a behaviour change.
"""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import _choice_probabilities, simulator
from ssms.config.model_config_builder import ModelConfigBuilder

OMISSION_SENTINEL = -999.0
KEYS = ["choice_p", "choice_p_no_omission", "omission_p", "nogo_p", "go_p"]


def _reference(rts, choices, possible_choices):
    """The original per-trial loop from ``simulator()``, verbatim in substance."""
    n_trials = rts.shape[1]
    x = {
        "choice_p": np.zeros((n_trials, len(possible_choices))),
        "choice_p_no_omission": np.zeros((n_trials, len(possible_choices))),
        "omission_p": np.zeros((n_trials, 1)),
        "nogo_p": np.zeros((n_trials, 1)),
        "go_p": np.zeros((n_trials, 1)),
    }
    for k in range(n_trials):
        out_len = rts[:, k, :].shape[0]
        out_len_no_omission = rts[:, k, :][rts[:, k, :] != OMISSION_SENTINEL].shape[0]

        for n, choice in enumerate(possible_choices):
            x["choice_p"][k, n] = (choices[:, k, :] == choice).sum() / out_len
            if out_len_no_omission > 0:
                x["choice_p_no_omission"][k, n] = (
                    choices[:, k, :][rts[:, k, :] != OMISSION_SENTINEL] == choice
                ).sum() / out_len_no_omission
            else:
                x["choice_p_no_omission"][k, n] = OMISSION_SENTINEL

        x["omission_p"][k, 0] = (rts[:, k, :] == OMISSION_SENTINEL).sum() / out_len
        x["nogo_p"][k, 0] = (
            (choices[:, k, :] != max(possible_choices))
            | (rts[:, k, :] == OMISSION_SENTINEL)
        ).sum() / out_len
        x["go_p"][k, 0] = 1 - x["nogo_p"][k, 0]
    return x


def _assert_identical(got, expected):
    for key in KEYS:
        assert got[key].dtype == expected[key].dtype, key
        assert got[key].shape == expected[key].shape, key
        np.testing.assert_array_equal(got[key], expected[key], err_msg=key)


def _synthetic(n_trials, n_samples, possible_choices, omission_rate, seed):
    rng = np.random.default_rng(seed)
    shape = (n_samples, n_trials, 1)
    choices = rng.choice(np.asarray(possible_choices), size=shape).astype(np.int32)
    rts = rng.uniform(0.1, 3.0, size=shape).astype(np.float32)
    rts[rng.random(shape) < omission_rate] = OMISSION_SENTINEL
    return rts, choices


@pytest.mark.parametrize("n_trials, n_samples", [(1, 1000), (50, 200), (2000, 1)])
@pytest.mark.parametrize("possible_choices", [[-1, 1], [0, 1, 2], [0, 1, 2, 3]])
@pytest.mark.parametrize("omission_rate", [0.0, 0.2])
def test_matches_reference_loop(n_trials, n_samples, possible_choices, omission_rate):
    rts, choices = _synthetic(n_trials, n_samples, possible_choices, omission_rate, 0)
    _assert_identical(
        _choice_probabilities(rts, choices, possible_choices),
        _reference(rts, choices, possible_choices),
    )


@pytest.mark.parametrize("possible_choices", [[-1, 1], [0, 1, 2, 3]])
def test_trial_with_only_omissions(possible_choices):
    rts, choices = _synthetic(5, 100, possible_choices, 0.3, 1)
    rts[:, 2, :] = OMISSION_SENTINEL
    got = _choice_probabilities(rts, choices, possible_choices)
    _assert_identical(got, _reference(rts, choices, possible_choices))
    assert np.all(got["choice_p_no_omission"][2] == OMISSION_SENTINEL)
    assert got["omission_p"][2, 0] == 1.0
    assert got["nogo_p"][2, 0] == 1.0


def test_possible_choices_as_array():
    """Cython simulators report ``possible_choices`` as lists or arrays."""
    possible_choices = np.arange(3)
    rts, choices = _synthetic(20, 50, possible_choices, 0.1, 2)
    _assert_identical(
        _choice_probabilities(rts, choices, possible_choices),
        _reference(rts, choices, possible_choices),
    )


def _unsqueeze(arr, n_trials, n_samples):
    """Undo the output-compatibility squeeze ``simulator()`` applies."""
    arr = np.asarray(arr)
    if n_trials == 1:
        return np.expand_dims(arr, axis=1)
    if n_samples == 1:
        return np.expand_dims(arr, axis=0)
    return arr


@pytest.mark.parametrize("model", ["ddm_deadline", "race_3", "lca_4"])
@pytest.mark.parametrize("n_trials, n_samples", [(1, 500), (20, 50), (300, 1)])
def test_simulator_end_to_end(model, n_trials, n_samples):
    # The probabilities are only computed by the ``simulator()`` function (the path
    # ``hssm_support`` uses); ``Simulator.simulate`` returns rts/choices/metadata.
    cfg = ModelConfigBuilder.from_model(model)
    theta = np.tile(np.array(cfg["default_params"], dtype=float), (n_trials, 1))
    if "deadline" in cfg["params"]:
        # A deadline short enough that a good share of responses are omitted
        theta[:, cfg["params"].index("deadline")] = 0.6
    out = simulator(theta, model=model, n_samples=n_samples, random_state=7)

    rts = _unsqueeze(out["rts"], n_trials, n_samples)
    choices = _unsqueeze(out["choices"], n_trials, n_samples)
    assert rts.shape == (n_samples, n_trials, 1)
    possible_choices = out["metadata"]["possible_choices"]
    _assert_identical(out, _reference(rts, choices, possible_choices))
    if model.endswith("_deadline") and n_trials * n_samples > 1:
        assert np.any(rts == OMISSION_SENTINEL)
