"""Define a collection of drift functions for the simulators in the package.

All drift functions accept `v` as a parameter and return the FINAL drift value,
consistent with boundary functions that accept `a` and return the final boundary value.
"""

# External
from collections.abc import Callable
from functools import partial

import numpy as np
from scipy.stats import norm


# TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B008, FIX002
def constant(
    t: np.ndarray = np.arange(0, 20, 0.1),  # noqa: B008
    v: float = 0.0,
) -> np.ndarray:
    """Constant drift function.

    Arguments
    ---------
        t: np.ndarray, optional
            Timepoints at which to evaluate the drift. Defaults to
            np.arange(0, 20, 0.1).
        v: float, optional
            Drift rate. Defaults to 0.0.

    Returns
    -------
        np.ndarray: Array of constant drift values = v, same length as t
    """
    return np.full(t.shape[0], v, dtype=np.float32)


def gamma_drift(
    t: np.ndarray = np.arange(  # noqa: B008
        0, 20, 0.1
    ),  # TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B008, FIX002
    v: float = 0.0,
    shape: float = 2,
    scale: float = 0.01,
    c: float = 1.5,
) -> np.ndarray:
    """Drift function that follows a scaled gamma distribution.

    The final drift is v + (scaled gamma component).

    Arguments
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        v: float
            Base drift rate. Defaults to 0.0.
        shape: float
            Shape parameter of the gamma distribution
        scale: float
            Scale parameter of the gamma distribution
        c: float
            Scalar parameter that scales the peak of
            the gamma distribution.
            (Note this function follows a gamma distribution
            but does not integrate to 1)

    Return
    ------
        np.ndarray
            The final drift (v + gamma component) evaluated at the supplied timepoints t.
    """
    num_ = np.power(t, shape - 1) * np.exp(np.divide(-t, scale))
    div_ = (
        np.power(shape - 1, shape - 1)
        * np.power(scale, shape - 1)
        * np.exp(-(shape - 1))
    )
    gamma_component = c * np.divide(num_, div_)
    return v + gamma_component


def ds_support_analytic(
    t: np.ndarray | None,
    init_p: float = 0,
    fix_point: float = 1,
    slope: float = 2,
) -> np.ndarray:
    """Solve DE.

    DE is of the form:
       x' = slope*(fix_point - x),
       with initial condition init_p.
       The solution takes the form:
       (init_p - fix_point) * exp(-slope * t) + fix_point

    Arguments
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift. Usually np.arange() of some sort.
        init_p: float
            Initial condition of dynamical system
        fix_point: float
            Fixed point of dynamical system
        slope: float
            Coefficient in exponent of the solution.
    Return
    ------
    np.ndarray
         The gamma drift evaluated at the supplied timepoints t.
    """
    if t is None:
        t = np.arange(0, 20, 0.1)

    return (init_p - fix_point) * np.exp(-(slope * t)) + fix_point


def conflict_ds_drift(
    t: np.ndarray | None,
    v: float = 0.0,
    tinit: float = 0,
    dinit: float = 0,
    tslope: float = 1,
    dslope: float = 1,
    tfixedp: float = 1,
    tcoh: float = 1.5,
    dcoh: float = 1.5,
) -> np.ndarray:
    """This drift is inspired by a conflict task which
       involves a target and a distractor stimuli both presented
       simultaneously.

       Two drift timecourses are linearly combined weighted
       by the coherence in the respective target and distractor stimuli.
       Each timecourse follows a dynamical system as described
       in the ds_support_analytic() function.

       The final drift is v + (combined drift timecourse).

    Arguments
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        v: float
            Base drift rate (typically 0 for conflict models). Defaults to 0.0.
        tinit: float
            Initial condition of target drift timecourse
        dinit: float
            Initial condition of distractor drift timecourse
        tslope: float
            Slope parameter for target drift timecourse
        dslope: float
            Slope parameter for distractor drift timecourse
        tfixedp: float
            Fixed point for target drift timecourse
        tcoh: float
            Coefficient for the target drift timecourse
        dcoh: float
            Coefficient for the distractor drift timecourse
    Return
    ------
    np.ndarray
         The full drift timecourse evaluated at the supplied timepoints t.
    """  # noqa: D205, D401, D404
    if t is None:
        t = np.arange(0, 20, 0.1)
    w_t = ds_support_analytic(t=t, init_p=tinit, fix_point=tfixedp, slope=tslope)

    w_d = ds_support_analytic(t=t, init_p=dinit, fix_point=0, slope=dslope)

    drift_component = (w_t * tcoh) + (w_d * dcoh)

    return v + drift_component


def attend_drift(
    t: np.ndarray = np.arange(  # noqa: B008
        0, 20, 0.1
    ),  # TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B008, FIX002
    v: float = 0.0,
    ptarget: float = -0.3,
    pouter: float = -0.3,
    pinner: float = 0.3,
    r: float = 0.5,
    sda: float = 2,
) -> np.ndarray:
    """Shrink spotlight model, which involves a time varying
    function dependent on a linearly decreasing standard deviation of attention.

    The final drift is v + (attention-weighted drift component).

    Arguments
    --------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        v: float
            Base drift rate (typically 0 for shrink_spot models). Defaults to 0.0.
        pouter: float
            perceptual input for outer flankers
        pinner: float
            perceptual input for inner flankers
        ptarget: float
            perceptual input for target flanker
        r: float
            rate parameter for sda decrease
        sda: float
            width of attentional spotlight
    Return
    ------
    np.ndarray
        Final drift evaluated at timepoints t
    """  # noqa: D205
    new_sda = np.maximum(sda - r * t, 0.001)

    a_outer = norm.sf(1.5, loc=0, scale=new_sda)
    a_inner = norm.cdf(1.5, loc=0, scale=new_sda) - norm.cdf(0.5, loc=0, scale=new_sda)
    a_target = norm.cdf(0.5, loc=0, scale=new_sda) - norm.cdf(
        -0.5, loc=0, scale=new_sda
    )

    drift_component = (
        (2 * pouter * a_outer) + (2 * pinner * a_inner) + (ptarget * a_target)
    )

    return v + drift_component


def attend_drift_simple(
    t: np.ndarray = np.arange(  # noqa: B008
        0, 20, 0.1
    ),  # TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B008, FIX002
    v: float = 0.0,
    ptarget: float = -0.3,
    pouter: float = -0.3,
    r: float = 0.5,
    sda: float = 2,
) -> np.ndarray:
    """Drift function for shrinking spotlight model, which involves a time varying
    function dependent on a linearly decreasing standard deviation of attention.

    The final drift is v + (attention-weighted drift component).

    Arguments
    --------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        v: float
            Base drift rate (typically 0 for shrink_spot models). Defaults to 0.0.
        pouter: float
            perceptual input for outer flankers
        ptarget: float
            perceptual input for target flanker
        r: float
            rate parameter for sda decrease
        sda: float
            width of attentional spotlight
    Return
    ------
    np.ndarray
        Final drift evaluated at timepoints t
    """  # noqa: D205
    new_sda = np.maximum(sda - r * t, 0.001)
    a_outer = 1.0 - norm.cdf(
        0.5, loc=0, scale=new_sda
    )  # equivalent to norm.sf(0.5, loc=0, scale=new_sda)
    a_target = norm.cdf(0.5, loc=0, scale=new_sda) - 0.5

    drift_component = (2 * pouter * a_outer) + (2 * ptarget * a_target)

    return v + drift_component


def stimflex_support(
    t: np.ndarray, onset: float, offset: float, coh: float
) -> np.ndarray:
    """
    Construct a rectangular coherence timecourse, with discrete and
    potentially variable onsets and offsets of stimulus evidence.

    Arguments
    ---------
        t: np.ndarray
            Timepoints within trial.
        onset: float
            Onset time of the coherence pulse.
        offset: float
            Offset time of the coherence pulse.
        coh: float
            Coherence of the stimulus when 'on'.
    Returns
    -------
        np.ndarray: Array of coherence values, same length as t.
    """
    cohs = np.zeros_like(t)
    cohs[(t >= onset) & (t <= offset)] = coh
    return cohs


def conflict_dsstimflex_drift(
    t: np.ndarray | None,
    v: float = 0.0,
    tinit: float = 0,
    dinit: float = 0,
    tslope: float = 1,
    dslope: float = 1,
    tfixedp: float = 1,
    tcoh: float = 1.0,
    dcoh: float = 1.0,
    tonset: float = 0,
    donset: float = 0,
    rel_first: bool = True,
) -> np.ndarray:
    """Drift function for conflict task with stimuli with potentially variable onset.

    The final drift is v + (combined drift timecourse).

    Arguments:
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        v: float
            Base drift rate (typically 0 for conflict models). Defaults to 0.0.
        tcoh: float
            Coherence of the target stimulus when 'on'.
        dcoh: float
            Coherence of the distractor stimulus when 'on'.
        tinit: float
            Initial condition of target drift timecourse.
        dinit: float
            Initial condition of distractor drift timecourse.
        tslope: float
            Slope parameter for target drift timecourse.
        dslope: float
            Slope parameter for distractor drift timecourse.
        tfixedp: float
            Fixed point for target drift timecourse.
        tonset: float
            Onset time of the target stimulus coherence.
        donset: float
            Onset time of the distractor stimulus coherence.
        rel_first: bool
            If True, the first stimulus to appear (target or distractor)
            is treated as appearing at time 0, and the other stimulus
            is adjusted accordingly. If False, the onsets are treated
            as absolute times.
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    if rel_first:
        first = min(tonset, donset)
        tonset -= first
        donset -= first
    offset = np.max(t)
    tcohs = stimflex_support(t, tonset, offset, tcoh)
    dcohs = stimflex_support(t, donset, offset, dcoh)

    w_t = ds_support_analytic(t=t, init_p=tinit, fix_point=tfixedp, slope=tslope)
    w_d = ds_support_analytic(t=t, init_p=dinit, fix_point=0, slope=dslope)
    drift_component = (w_t * tcohs) + (w_d * dcohs)

    return v + drift_component


def conflict_stimflex_drift(
    t: np.ndarray | None,
    v: float = 0.0,
    vt: float = 0,
    vd: float = 0,
    tcoh: float = 1.0,
    dcoh: float = 1.0,
    tonset: float = 0,
    donset: float = 0,
    toffset: float | None = None,
    doffset: float | None = None,
    rel_first: bool = False,
    sum_drifts: bool = True,
) -> np.ndarray:
    """Drift function for conflict task with stimuli with potentially variable onset and duration.

    The final drift is v + (combined drift timecourse) when sum_drifts=True.
    When sum_drifts=False, returns 2D array for dual-drift models (v not added).

    Arguments:
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        v: float
            Base drift rate (typically 0 for conflict models). Defaults to 0.0.
            Only used when sum_drifts=True.
        tcoh: float
            Coherence of the target stimulus when 'on'.
        dcoh: float
            Coherence of the distractor stimulus when 'on'.
        vt: float
            Static drift-rate of target stimulus, when 'on'.
        vd: float
            Static drift-rate of distractor stimulus, when 'on'.
        tonset: float
            Onset time of the target stimulus coherence.
        donset: float
            Onset time of the distractor stimulus coherence.
        toffset, doffset: float or None
            Duration of the stimulus coherence pulse. If None, the pulse
            lasts until the end of the trial.
        rel_first: bool
            If True, the first stimulus to appear (target or distractor)
            is treated as appearing at time 0, and the other stimulus
            is adjusted accordingly. If False, the onsets are treated
            as absolute times.
        sum_drifts: bool
            If True, the drift contributions from target and distractor
            are summed to produce a single drift timecourse. If False,
            a 2D array is returned with separate columns for target
            and distractor drift timecourses (for dual-drift models).
    Returns
    -------
        np.ndarray: Array of drift values, same length as t. If sum_drifts
            is False, the array has shape (len(t), 2)
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    if rel_first:
        first = min(tonset, donset)
        tonset -= first
        donset -= first
    if toffset is None:
        toffset = np.max(t)
    if doffset is None:
        doffset = np.max(t)
    tcohs = stimflex_support(t, tonset, toffset, tcoh)
    dcohs = stimflex_support(t, donset, doffset, dcoh)
    if sum_drifts:
        drift_component = vt * tcohs + vd * dcohs
        return v + drift_component
    else:
        # Dual-drift mode: return 2D array (v not added - handled differently)
        return np.column_stack((vt * tcohs, vd * dcohs))


# Re-use drift fun but set different default args
conflict_stimflexrel1_drift = partial(conflict_stimflex_drift, rel_first=True)
conflict_stimflexrel1_dual_drift = partial(
    conflict_stimflex_drift, rel_first=True, sum_drifts=False
)


# Type alias for drift functions
DriftFunction = Callable[..., np.ndarray]

attend_drift: DriftFunction = attend_drift  # noqa: PLW0127
constant: DriftFunction = constant  # noqa: PLW0127
gamma_drift: DriftFunction = gamma_drift  # noqa: PLW0127
ds_support_analytic: DriftFunction = ds_support_analytic  # noqa: PLW0127
conflict_ds_drift: DriftFunction = conflict_ds_drift  # noqa: PLW0127
conflict_dsstimflex_drift: DriftFunction = conflict_dsstimflex_drift  # noqa: PLW0127
conflict_stimflex_drift: DriftFunction = conflict_stimflex_drift  # noqa: PLW0127
