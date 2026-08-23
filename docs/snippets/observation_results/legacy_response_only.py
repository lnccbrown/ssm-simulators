import numpy as np

from ssms.basic_simulators import OMISSION_SENTINEL, normalize_simulator_result


legacy_result = {
    # n_samples=1 and n_trials=3 historically produces shape (3, 1).
    "rts": np.full((3, 1), -1.0, dtype=np.float32),
    "choices": np.array([[0], [1], [OMISSION_SENTINEL]], dtype=np.int16),
    "metadata": {
        "simulator": "choice_only_rlssm",
        "possible_choices": (0, 1, 2),
        "placeholder_rt": -1.0,
    },
}

validated_result = normalize_simulator_result(
    legacy_result,
    expected_n_samples=1,
    expected_n_trials=3,
    observation_schema=(
        {
            "name": "response",
            "kind": "categorical",
            "values": (0, 1, 2),
        },
    ),
    source_projection=(("choices", "response"),),
)

assert validated_result["observations"].shape == (1, 3, 1)
assert validated_result["observations"].dtype == np.float64
assert validated_result["omission_mask"].tolist() == [[False, False, True]]
assert np.all(validated_result["rts"] == -1.0)
assert validated_result["rts"] is legacy_result["rts"]
