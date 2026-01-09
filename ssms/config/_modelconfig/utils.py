"""
Utility functions for model configuration.
"""


def _new_param(default: float, lower: float, upper: float) -> dict:
    return {"default": default, "bounds": [lower, upper]}


def _get(params: dict, field: str):
    if field == "name":
        return list(params.keys())
    elif field == "defaults":
        return [param["default"] for param in params.values()]
    elif field == "bounds":
        lower = [param["bounds"][0] for param in params.values()]
        upper = [param["bounds"][1] for param in params.values()]
        return [lower, upper]
    else:
        raise ValueError(f"Unknown field: {field}")


def _new_config(
    name,
    param_dict,
    boundary_name,
    boundary,
    drift_name,
    drift_fun,
    choices,
    n_particles,
    simulator,
    sampling_transforms=None,
    simulation_transforms=None,
):
    return {
        "name": name,
        "params": list(param_dict.keys()),
        "param_bounds": _get(param_dict, "bounds"),
        "boundary_name": boundary_name,
        "boundary": boundary,
        "drift_name": drift_name,
        "drift_fun": drift_fun,
        "n_params": len(param_dict),
        "default_params": _get(param_dict, "defaults"),
        "nchoices": len(choices),
        "choices": choices,
        "n_particles": n_particles,
        "simulator": simulator,
        "parameter_transforms": {
            "sampling": sampling_transforms or [],
            "simulation": simulation_transforms or [],
        },
    }
