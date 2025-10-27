"""Configuration for conflict models with dynamical drift."""

import cssm
from ssms.basic_simulators import boundary_functions as bf, drift_functions as df
from ssms.config._modelconfig.utils import _new_config, _new_param


def get_conflict_ds_config():
    return _new_config(
        name="conflict_ds",
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
        drift_name="conflict_ds_drift",
        drift_fun=df.conflict_ds_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex,
    )


def get_conflict_ds_angle_config():
    return _new_config(
        name="conflict_ds_angle",
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
        drift_name="conflict_ds_drift",
        drift_fun=df.conflict_ds_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex,
    )


def get_conflict_dsstimflex_config():
    return _new_config(
        name="conflict_dsstimflex",
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
        drift_name="conflict_dsstimflex_drift",
        drift_fun=df.conflict_dsstimflex_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex,
    )


def get_conflict_dsstimflex_angle_config():
    return _new_config(
        name="conflict_dsstimflex_angle",
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
        drift_name="conflict_dsstimflex_drift",
        drift_fun=df.conflict_dsstimflex_drift,
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
            vt=_new_param(2.0, 0.0, 5.0),
            vd=_new_param(2.0, 0.0, 5.0),
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
            vt=_new_param(2.0, 0.0, 5.0),
            vd=_new_param(2.0, 0.0, 5.0),
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
            vt=_new_param(2.0, 0.0, 5.0),
            vd=_new_param(2.0, 0.0, 5.0),
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
            vt=_new_param(2.0, 0.0, 5.0),
            vd=_new_param(2.0, 0.0, 5.0),
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
            vt=_new_param(2.0, 0.0, 5.0),
            vd=_new_param(2.0, 0.0, 5.0),
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
            vt=_new_param(2.0, 0.0, 5.0),
            vd=_new_param(2.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
            tonset=_new_param(0.0, 0.0, 1.0),
            donset=_new_param(0.0, 0.0, 1.0),
            toffset=_new_param(0.2, 0.0, 1.0),
            doffset=_new_param(0.2, 0.0, 1.0),
            gt=_new_param(0.0, 0.0, 1.0),
            gd=_new_param(0.0, 0.0, 1.0),
        ),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_stimflexrel1_dual_drift",
        drift_fun=df.conflict_stimflexrel1_dual_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_leak2,
    )
