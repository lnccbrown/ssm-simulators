import numpy as np

from ssms.basic_simulators import validate_observation_result


boundary = np.linspace(1.0, 0.0, 100)
result = {
    "observations": np.array([[[0.42, 0.8], [0.71, 0.35]]], dtype=np.float64),
    "omission_mask": np.array([[False, False]], dtype=bool),
    "metadata": {
        "observation_schema_version": 1,
        "observation_schema": (
            {
                "name": "rt",
                "kind": "continuous",
                "lower": 0.0,
                "lower_inclusive": False,
            },
            {
                "name": "confidence",
                "kind": "continuous",
                "lower": 0.0,
                "upper": 1.0,
            },
        ),
        "simulator": "example_confidence_model",
        "boundary": boundary,
    },
}

validated_result = validate_observation_result(result)

assert validated_result["observations"].shape == (1, 2, 2)
assert validated_result["metadata"]["boundary"] is boundary
