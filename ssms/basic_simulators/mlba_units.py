"""Convert the paper's MLBA parameterisation to the library's LBA units.

The paper fixes the drift SD at ``s = 1`` (Table 2); ``ssms`` LBA-type models
default to ``sigma_noise = 0.1``. Scaling attributes, ``I0``, ``A``, ``chi``
and ``s`` by ``c`` and the decay constants by ``1 / c`` leaves every RT and
choice unchanged (see the ``mlba_frontend`` docstring), so with
``c = sigma_noise / s`` the paper's model runs under the library default noise.
"""

from __future__ import annotations

LIBRARY_SIGMA_NOISE = 0.1


def paper_to_library(paper: dict, options, sigma_noise: float = LIBRARY_SIGMA_NOISE):
    """Return an ``mlba{n}`` theta dict (library units) for the paper's values.

    ``paper`` keys: m, lambda1, lambda2, I0, A, chi, s, and optionally beta, t0.
    ``options``: sequence of (P, Q) on the paper's attribute scale.
    """
    c = sigma_noise / float(paper.get("s", 1.0))
    theta = {
        "m": float(paper["m"]),
        "lambda1": float(paper["lambda1"]) / c,
        "lambda2": float(paper["lambda2"]) / c,
        "I0": float(paper["I0"]) * c,
        "beta": float(paper.get("beta", 1.0)),
        "A": float(paper["A"]) * c,
        "b": float(paper["chi"]) * c,
        "t": float(paper.get("t0", 0.0)),
    }
    for i, (p, q) in enumerate(options):
        theta[f"P{i}"] = float(p) * c
        theta[f"Q{i}"] = float(q) * c
    return theta
