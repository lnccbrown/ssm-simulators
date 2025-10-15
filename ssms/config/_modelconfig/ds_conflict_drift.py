"""Configuration for conflict models with dynamical drift."""

import cssm
from ssms.basic_simulators import boundary_functions as bf, drift_functions as df


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
    }


def get_ds_conflict_drift_config():
    return _new_config(
        name="ds_conflict_drift",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            tinit=_new_param(2.0, 0.0, 5.0),
            dinit=_new_param(2.0, 0.0, 5.0),
            tslope=_new_param(2.0, 0.01, 5.0),
            dslope=_new_param(2.0, 0.01, 5.0),
            tfixedp=_new_param(3.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
        ),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="ds_conflict_drift",
        drift_fun=df.ds_conflict_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex,
    )


def get_ds_conflict_drift_angle_config():
    return _new_config(
        name="ds_conflict_drift_angle",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            tinit=_new_param(2.0, 0.0, 5.0),
            dinit=_new_param(2.0, 0.0, 5.0),
            tslope=_new_param(2.0, 0.01, 5.0),
            dslope=_new_param(2.0, 0.01, 5.0),
            tfixedp=_new_param(3.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
            theta=_new_param(0.0, 0.0, 1.3),
        ),
        boundary_name="angle",
        boundary=bf.angle,
        drift_name="ds_conflict_drift",
        drift_fun=df.ds_conflict_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex,
    )


def get_ds_conflict_stimflex_config():
    return _new_config(
        name="ds_conflict_stimflex",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            tinit=_new_param(2.0, 0.0, 5.0),
            dinit=_new_param(2.0, 0.0, 5.0),
            tslope=_new_param(2.0, 0.01, 5.0),
            dslope=_new_param(2.0, 0.01, 5.0),
            tfixedp=_new_param(3.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
            tonset=_new_param(0.0, 0.0, 1.0),
            donset=_new_param(0.0, 0.0, 1.0),
        ),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="ds_conflict_stimflex_drift",
        drift_fun=df.ds_conflict_stimflex_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex,
    )


def get_ds_conflict_stimflex_angle_config():
    return _new_config(
        name="ds_conflict_stimflex_angle",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            tinit=_new_param(2.0, 0.0, 5.0),
            dinit=_new_param(2.0, 0.0, 5.0),
            tslope=_new_param(2.0, 0.01, 5.0),
            dslope=_new_param(2.0, 0.01, 5.0),
            tfixedp=_new_param(3.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
            tonset=_new_param(0.0, 0.0, 1.0),
            donset=_new_param(0.0, 0.0, 1.0),
            theta=_new_param(0.0, 0.0, 1.3),
        ),
        boundary_name="angle",
        boundary=bf.angle,
        drift_name="ds_conflict_stimflex_drift",
        drift_fun=df.ds_conflict_stimflex_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex,
    )


def get_conflict_stimflex_config():
    return _new_config(
        name="conflict_stimflex",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            v_t=_new_param(2.0, 0.0, 5.0),
            v_d=_new_param(2.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
            tonset=_new_param(0.0, 0.0, 1.0),
            donset=_new_param(0.0, 0.0, 1.0),
        ),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_stimflex_drift",
        drift_fun=df.conflict_stimflex_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex,
    )


def get_conflict_stimflex_angle_config():
    return _new_config(
        name="conflict_stimflex_angle",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            v_t=_new_param(2.0, 0.0, 5.0),
            v_d=_new_param(2.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
            tonset=_new_param(0.0, 0.0, 1.0),
            donset=_new_param(0.0, 0.0, 1.0),
            theta=_new_param(0.0, 0.0, 1.3),
        ),
        boundary_name="angle",
        boundary=bf.angle,
        drift_name="conflict_stimflex_drift",
        drift_fun=df.conflict_stimflex_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex,
    )


def get_conflict_stimflexrel1_config():
    return _new_config(
        name="conflict_stimflexrel1",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            v_t=_new_param(2.0, 0.0, 5.0),
            v_d=_new_param(2.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
            tonset=_new_param(0.0, 0.0, 1.0),
            donset=_new_param(0.0, 0.0, 1.0),
        ),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_stimflexrel1_drift",
        drift_fun=df.conflict_stimflexrel1_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex,
    )


def get_conflict_stimflexrel1_angle_config():
    return _new_config(
        name="conflict_stimflexrel1_angle",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            v_t=_new_param(2.0, 0.0, 5.0),
            v_d=_new_param(2.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
            tonset=_new_param(0.0, 0.0, 1.0),
            donset=_new_param(0.0, 0.0, 1.0),
            theta=_new_param(0.0, 0.0, 1.3),
        ),
        boundary_name="angle",
        boundary=bf.angle,
        drift_name="conflict_stimflexrel1_drift",
        drift_fun=df.conflict_stimflexrel1_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex,
    )


def get_conflict_stimflexrel1_leak_config():
    return _new_config(
        name="conflict_stimflexrel1_leak",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            v_t=_new_param(2.0, 0.0, 5.0),
            v_d=_new_param(2.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
            tonset=_new_param(0.0, 0.0, 1.0),
            donset=_new_param(0.0, 0.0, 1.0),
            toffset=_new_param(0.2, 0.0, 1.0),
            doffset=_new_param(0.2, 0.0, 1.0),
            g=_new_param(0.0, 0.0, 1.0),
        ),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_stimflexrel1_drift",
        drift_fun=df.conflict_stimflexrel1_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_leak,
    )


def get_conflict_stimflexrel1_leak2_config():
    return _new_config(
        name="conflict_stimflexrel1_leak2",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            v_t=_new_param(2.0, 0.0, 5.0),
            v_d=_new_param(2.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
            tonset=_new_param(0.0, 0.0, 1.0),
            donset=_new_param(0.0, 0.0, 1.0),
            toffset=_new_param(0.2, 0.0, 1.0),
            doffset=_new_param(0.2, 0.0, 1.0),
            g_t=_new_param(0.0, 0.0, 1.0),
            g_d=_new_param(0.0, 0.0, 1.0),
        ),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_stimflexrel1_dual_drift",
        drift_fun=df.conflict_stimflexrel1_dual_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_leak2,
    )
