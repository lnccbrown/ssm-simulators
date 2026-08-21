# Contributed from a paper by HSSMCortex.
#
#   Paper   : fontanesi-2019-rlddm
#   Spec    : papers/ssm-theory/fontanesi-2019-rlddm.spec.md
#   Report  : papers/ssm-theory/fontanesi-2019-rlddm.report.json
#   ssms    : 0.13.2
#   Verdict : draft (amber)
#
# Machine-generated from the paper's ModelSpec and checked against the verification
# ladder described in the pull request body. This file is the artifact that report was
# produced from; it is transferred rather than rewritten, so the two cannot drift apart.

"""GENERATED from papers/ssm-theory/fontanesi-2019-rlddm.md — do not edit by hand.

Emitted by ``scripts/generate_model.py`` from ``dev_rlddm``'s ModelSpec. Every
component below either resolves in an ``ssms`` registry or is generated from the spec's own
expression, and the report says which.

Regenerate rather than patch: an edit here is invisible to the spec that is supposed to describe it.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

#: ASSUMED -- the paper states none of these, and each was filled from the source
#: named beside it. A reviewer is being asked to accept them; the verdict is capped
#: at a draft while any remain.
#:   model_name = 'dev_rlddm'  <- the spec's own `family`
MODEL_NAME = "dev_rlddm"
DONOR = "conflict_ds"
#: Fields the paper left silent, filled from the source named beside each. Empty for a spec that
#: stated everything codegen reads.
ASSUMPTIONS: list[dict[str, Any]] = [{"field": "model_name", "value": "dev_rlddm", "source": "the spec's own `family`", "why": "`ssms` carries 113 entries and none of them cites a paper; the convention is compositional. `rlddm` is the paper's own name for the model class, taken from `family`, and `dev_` is where a machine-proposed model belongs until a maintainer promotes it -- the namespace `dev_rlwm_lba_pw_v1` already occupies", "kind": "notational", "falsifier": "a maintainer preferring a different token, or an existing entry that already models this -- in which case this paper contributes a citation and a benchmark to it rather than `dev_rlddm`", "confidence": "medium"}]
BASE_THETA: dict[str, float] = {"v": 0.0, "a": 0.0, "z": 0.5, "t": 0.75, "etaPos": 0.155, "etaNeg": 0.155, "vMod": 1.05, "vMax": 3.75, "aFix": 1.05, "aMod": -0.02, "qCor": 41.25, "qInc": 41.25, "qPres": 41.25}
REDUCTION_THETA: dict[str, float] = {"etaPos": 0.0, "etaNeg": 0.0, "vMod": 0.0, "vMax": 0.0, "aFix": 0.0, "aMod": 0.0, "qCor": 0.0, "qInc": 0.0, "qPres": 0.0}
GENERATED_COMPONENTS = ["boundary", "drift"]

#: The spec's own parameter names, mapped onto the ones theta actually carries. Emitted because
#: the spec's `relations.reduces_to.parameter_map`, its benchmarks and its `design.schedule` binds
#: are all written in the PAPER's notation while this config is written in the registry's, and the
#: rungs have to evaluate one against the other.
#:
#: Two renamings are merged here and both must be, because a consumer cannot tell them apart and
#: should not have to: the conventions the spec states about itself (`t_er` is the non-decision
#: time, so it is `t`), and the spelling `ssms` requires of anything it registers (`eta_pos`
#: carries an underscore, which its import-time validator rejects, so it is `etaPos`).
# Every map from a name the SPEC uses to the name this module emits, in one dict. Three produce
# one: the convention aliases, the DYNAMICS renaming when the drift was inherited rather than
# generated, and the upstream renaming `ssms` imposes.
#
# The dynamics renaming was missing, and the omission was invisible until a paper needed it.
# `busemeyer-1993-dft` writes its drift `delta - sPlusC * P`, which `ornstein` already computes as
# `v - g * x`, so the module is built on `v` and `g` while the spec's reduction still names
# `delta`. Rung 2 evaluated the reduction`s parameter_map and came back `name 'delta' is not
# defined`, so the one rung that could have judged this model never ran -- reported as though the
# reduction were untestable rather than as a renaming this file failed to publish.
CONVENTION_ALIASES: dict[str, str] = {"t_er": "t", "eta_pos": "etaPos", "eta_neg": "etaNeg", "v_mod": "vMod", "v_max": "vMax", "a_fix": "aFix", "a_mod": "aMod", "q_cor": "qCor", "q_inc": "qInc", "q_pres": "qPres"}

#: Values derived from the spec rather than fitted -- a starting point stated as a fraction of the
#: boundary is a constant, not a parameter, and must not be given a range to be searched over.
DERIVED_CONSTANTS: dict[str, float] = {"z": 0.5}

def _donor_config() -> dict[str, Any]:
    """The donor's config, resolved without importing a package that is still initializing.

    `ssms` calls every registered factory from `_validate_configs()` at the END of
    `ssms/config/_modelconfig/__init__.py`, which itself runs while `ssms/config/__init__.py` is
    only part way through its own imports. So `from ssms.config import model_config` here raises
    `cannot import name 'model_config' from partially initialized module` -- and the whole package
    stops importing, for every user of it, on the strength of one contributed model.

    By the time the factory is called, every donor factory is already bound in the `_modelconfig`
    namespace, so it is taken from there. The package-level lookup remains as the path for any
    caller reaching this module after `ssms` has finished importing.
    """
    import importlib

    package = importlib.import_module("ssms.config._modelconfig")
    factory = getattr(package, f"get_{{DONOR}}_config", None)
    if factory is not None:
        return factory()
    from ssms.config import model_config

    return model_config[DONOR]


BOUNDARY_NAME = "dev_rlddm_boundary"

#: Parameters the boundary function expects. ``a`` must be here when the expression uses it: ssms
#: filters theta down to these names, so omitting one does not raise -- the function silently falls
#: back to its own default and simulates a bound unrelated to the one being passed.
BOUNDARY_PARAMS = ["aFix", "aMod", "qPres"]


def dev_rlddm_boundary(t: float | np.ndarray = 0.0, aFix: float = 1.0, aMod: float = 1.0, qPres: float = 1.0) -> float | np.ndarray:
    """Generated from the spec's boundary expression.

    ``b(t) = (exp(aFix + aMod * qPres)) * 0.5``

    This is the one place free-form code is written rather than a registry component named, and it
    is the capability a genuinely novel model is most likely to need.
    """
    t = np.asarray(t)
    return np.exp(aFix + aMod * qPres) * 0.5


def register_boundary_once() -> None:
    # From the SUBMODULE, not the package. This runs while `ssms.config` is still executing its own
    # `__init__` -- `_validate_configs()` calls every registered factory at import -- so importing
    # the package here raises `cannot import name ... from partially initialized module`.
    # `boundary_registry` is imported before `_modelconfig`, so reaching it directly is safe.
    from ssms.config.boundary_registry import get_boundary_registry, register_boundary

    if get_boundary_registry().is_registered(BOUNDARY_NAME):
        return
    register_boundary(name=BOUNDARY_NAME, function=dev_rlddm_boundary, params=BOUNDARY_PARAMS)


DRIFT_NAME = "dev_rlddm_drift"

#: Parameters the drift function expects. The same trap as BOUNDARY_PARAMS, and sharper: ssms
#: resolves the drift by NAME out of the registry and filters theta down to the names recorded
#: here, so a parameter omitted from this list does not raise -- the function falls back to its own
#: default and integrates a rate unrelated to the one being fitted.
DRIFT_PARAMS = ["vMod", "vMax", "qCor", "qInc"]


def dev_rlddm_drift(t: float | np.ndarray = 0.0, vMod: float = 1.0, vMax: float = 1.0, qCor: float = 1.0, qInc: float = 1.0) -> np.ndarray:
    """Generated from the spec's drift expression.

    ``v(t) = 2 * vMax / (1 + exp(-(vMod * (qCor - qInc)))) - vMax``

    Returns an ARRAY shaped like `t`, always, which is the contract every drift in the registry
    keeps: `constant`, `gamma_drift` and `attend_drift_simple` all annotate `np.ndarray` and all
    return one shaped like the time grid they are handed.

    Measured, so the reason is not overstated: `ssms` 0.13.2 also simulates a drift that returns a
    bare scalar. Matching the registry is a contract decision, not a workaround for a failure --
    and `np.zeros_like(t) +` costs nothing for an expression that already depends on `t`.
    """
    t = np.asarray(t, dtype=float)
    return np.zeros_like(t) + (2 * vMax / (1 + np.exp(-(vMod * (qCor - qInc)))) - vMax)


def register_drift_once() -> None:
    # From the SUBMODULE; see `register_boundary_once`.
    from ssms.config.drift_registry import get_drift_registry, register_drift

    if get_drift_registry().is_registered(DRIFT_NAME):
        return
    register_drift(name=DRIFT_NAME, function=dev_rlddm_drift, params=DRIFT_PARAMS)


def get_dev_rlddm_config() -> dict[str, Any]:
    """Return the generated ``model_config`` entry.

    The donor supplies the simulator and the parameter transforms — 80 of the zoo's 113 models
    carry a non-empty transform, so emitting one is the norm rather than an edge case — and this
    config swaps in the components the spec names.
    """
    from ssms.config.boundary_registry import get_boundary_registry

    donor = _donor_config()
    cfg = copy.deepcopy({
        k: v for k, v in donor.items()
        if k not in ("boundary", "simulator", "parameter_transforms", "param_bounds_dict")
    })

    cfg["name"] = MODEL_NAME
    cfg["params"] = ["v", "a", "z", "t", "etaPos", "etaNeg", "vMod", "vMax", "aFix", "aMod", "qCor", "qInc", "qPres"]
    cfg["param_bounds"] = [[0.0, 0.0, 0.5, 0.4, 0.01, 0.01, 0.1, 1.5, 0.3, -0.05, 27.5, 27.5, 27.5], [0.0, 0.0, 0.5, 1.1, 0.3, 0.3, 2.0, 6.0, 1.8, 0.01, 55.0, 55.0, 55.0]]
    cfg["n_params"] = 13
    cfg["default_params"] = [0.0, 0.0, 0.5, 0.75, 0.155, 0.155, 1.05, 3.75, 1.05, -0.02, 41.25, 41.25, 41.25]
    cfg["nchoices"] = 2
    cfg["choices"] = [-1, 1]
    cfg["n_particles"] = donor.get("n_particles", 1)
    # Carried so an oracle can tell a RENAMED parameter from a dead one: the spec still declares
    # `t_er` while the model is fitted over `t`, and perturbing the spec's name moves nothing.
    cfg["convention_aliases"] = CONVENTION_ALIASES
    cfg["derived_constants"] = DERIVED_CONSTANTS

    register_boundary_once()
    cfg["boundary_name"] = BOUNDARY_NAME
    cfg["boundary"] = dev_rlddm_boundary
    cfg["boundary_params"] = BOUNDARY_PARAMS
    register_drift_once()
    cfg["drift_name"] = DRIFT_NAME
    cfg["drift_fun"] = dev_rlddm_drift

    cfg["simulator"] = donor["simulator"]
    cfg["parameter_transforms"] = copy.deepcopy(donor["parameter_transforms"])
    return cfg
