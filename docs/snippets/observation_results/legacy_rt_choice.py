import numpy as np

from ssms.basic_simulators import OMISSION_SENTINEL, normalize_simulator_result


boundary = np.linspace(1.0, 0.0, 100)
legacy_result = {
    # n_samples=2 and n_trials=1 historically produces shape (2, 1).
    "rts": np.array([[0.42], [OMISSION_SENTINEL]], dtype=np.float32),
    "choices": np.array([[1.0], [OMISSION_SENTINEL]], dtype=np.float32),
    "metadata": {
        "simulator": "ddm",
        "possible_choices": (-1, 1),
        "boundary": boundary,
    },
}

validated_result = normalize_simulator_result(
    legacy_result,
    expected_n_samples=2,
    expected_n_trials=1,
    observation_schema=(
        {
            "name": "rt",
            "kind": "continuous",
            "lower": 0.0,
            "lower_inclusive": False,
        },
        {
            "name": "response",
            "kind": "categorical",
            "values": (-1, 1),
        },
    ),
    source_projection=(("rts", "rt"), ("choices", "response")),
)

assert validated_result["observations"].shape == (2, 1, 2)
assert validated_result["omission_mask"].tolist() == [[False], [True]]
assert np.isnan(validated_result["observations"][1, 0]).all()
assert validated_result["rts"] is legacy_result["rts"]
assert validated_result["metadata"]["boundary"] is boundary
assert legacy_result["rts"][1, 0] == OMISSION_SENTINEL
