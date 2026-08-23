import numpy as np

from ssms.basic_simulators import validate_observation_result


result = {
    "observations": np.array(
        [[[0.42, 0.8, -1.2], [0.71, 0.35, 2.4]]], dtype=np.float64
    ),
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
            {
                "name": "azimuth",
                "kind": "circular",
                "lower": -np.pi,
                "upper": np.pi,
            },
        ),
        "simulator": "example_multi_field_model",
    },
}

validated_result = validate_observation_result(result)

assert validated_result["observations"].shape == (1, 2, 3)
assert validated_result["omission_mask"].tolist() == [[False, False]]
