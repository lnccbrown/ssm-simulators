"""Shared fixtures for the Timed Racing Diffusion Model (TRDM) tests.

The TRDM configs (``trdm_2``, ``trdm_2_guess``, ``trdm_3``) are registered
statically through ``ssms/config/_modelconfig/__init__.py``; no import-time
registration is needed here.

``declarations`` reproduces the model part of the original contribution's
``declarations.yaml`` (names, bounds, defaults, neutral values and the declared
``trdm_3 -> racing_diffusion_3`` reduction) so that the registered configs are
checked against the declared contract.

``bundle_declarations`` / ``TRDM_BUNDLE_DIR`` are only used by the optional
paper-reproduction checks in ``test_trdm_bridge_reproduce.py``; they need the
original bundle's ``bridge.py``, ``reproduce.py`` and ``declarations.yaml``,
which are not part of the library.
"""

import os
import pathlib
import sys

import pytest

_NO_TRANSFORMS = {"sampling": [], "simulation": []}


def _p(name, lo, hi, default, neutral=None):
    return {"name": name, "bounds": [lo, hi], "default": default, "neutral": neutral}


_TRDM_2_PARAMETERS = [
    _p("v0", -3.0, 7.0, 2.0),
    _p("v1", -3.0, 7.0, 0.5),
    _p("b0", 0.2, 4.0, 1.0),
    _p("b1", 0.2, 4.0, 1.0),
    _p("t", 0.0, 2.0, 0.3),
    _p("vT", 0.0, 5.0, 1.0, 0.0),
    _p("sT", 0.0, 2.0, 0.5, 0.0),
    _p("tT", 0.0, 1.0, 0.05),
]

MODEL_DECLARATIONS = {
    "models": [
        {
            "name": "trdm_2",
            "base_model": None,
            "parameter_transforms": _NO_TRANSFORMS,
            "parameters": _TRDM_2_PARAMETERS,
            "reductions": [],
        },
        {
            "name": "trdm_2_guess",
            "base_model": None,
            "parameter_transforms": _NO_TRANSFORMS,
            "parameters": _TRDM_2_PARAMETERS + [_p("g", 0.0, 1.0, 0.5, 0.5)],
            "reductions": [],
        },
        {
            "name": "trdm_3",
            "base_model": "racing_diffusion_3",
            "parameter_transforms": _NO_TRANSFORMS,
            "parameters": [
                _p("v0", 0.0, 2.5, 1.0),
                _p("v1", 0.0, 2.5, 1.0),
                _p("v2", 0.0, 2.5, 1.0),
                _p("A", 0.0, 1.0, 0.5, 0.0),
                _p("b", 0.5, 3.0, 1.5),
                _p("t", 0.0, 2.0, 0.001),
                _p("vT", 0.0, 5.0, 0.5, 0.0),
                _p("sT", 0.0, 2.0, 0.3, 0.0),
                _p("tT", 0.0, 1.0, 0.0),
            ],
            "reductions": [
                {
                    "to": "racing_diffusion_3",
                    "settings": {"vT": 0.0, "sT": 0.0, "tT": 0.0},
                    "map": {
                        "v0": "v0",
                        "v1": "v1",
                        "v2": "v2",
                        "A": "A",
                        "b": "b",
                        "t": "t",
                    },
                    "expect": "distribution",
                }
            ],
        },
    ],
}


def bundle_dir():
    """Return the original contribution directory, or ``None`` when not set."""
    value = os.environ.get("TRDM_BUNDLE_DIR")
    if not value:
        return None
    path = pathlib.Path(value).resolve()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


@pytest.fixture(scope="session")
def declarations():
    return MODEL_DECLARATIONS


@pytest.fixture(scope="session")
def bundle_declarations():
    path = bundle_dir()
    if path is None or not (path / "declarations.yaml").exists():
        pytest.skip("TRDM_BUNDLE_DIR with declarations.yaml not set")
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((path / "declarations.yaml").read_text())
