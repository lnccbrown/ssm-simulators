import numpy as np

from ssms.basic_simulators import validate_observation_result


result = {
    "observations": np.array([[[1.0], [np.nan], [2.0]]], dtype=np.float64),
    "omission_mask": np.array([[False, True, False]], dtype=bool),
    "metadata": {
        "observation_schema_version": 1,
        "observation_schema": (
            {
                "name": "response",
                "kind": "categorical",
                "values": (0, 1, 2),
            },
        ),
        "simulator": "example_choice_only_model",
    },
}

validated_result = validate_observation_result(result)

assert validated_result["observations"].shape == (1, 3, 1)
assert validated_result["omission_mask"].tolist() == [[False, True, False]]
