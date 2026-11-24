"""Multiple-Threshold Race (MTR) model configuration.

Simulator: ``ssms.basic_simulators.mtr.multiple_threshold_race``. Registered
statically as ``"mtr"`` in ``get_model_config()`` (``mtr_deadline`` is derived
by the library's deadline-variant mechanism).

Description
-----------
Reynolds et al.'s Multiple-Threshold Race adds a "don't-know" response to a
two-accumulator Linear Ballistic Accumulator. Each accumulator has a choice
threshold ``b_i`` and a lower don't-know threshold ``d_i``. The first
accumulator to reach its choice threshold sets the RT; the response is that
accumulator's definitive choice if the losing accumulator is still below its
don't-know threshold, and "don't know" otherwise (Vickers' balance of
evidence). Rates are normal truncated to positive values; start points are
uniform on ``[0, A]``.

Parameters
----------
A : start-point range (shared). [0, 3]
B0, B1 : ``b_i - A`` (paper's B). [0.1, 5]
D0, D1 : ``d_i / b_i`` (paper's D = 1 - DK). [0, 1]; 1 = no don't-know.
v0, v1 : mean rates. [0, 6]
sv0, sv1 : rate SDs. [0.1, 3]
t : non-decision time (s). [0, 2]

Choices: 0 / 1 = definitive response of accumulator 0 / 1; 2 = don't know.

See Also
--------
lba2 : two-accumulator LBA (folded-normal rates, ``sd`` from ``sigma_noise``,
    ``t`` fixed at 0) -- the family this model extends. The MTR at ``D0 = D1 =
    1`` is a two-accumulator LBA with *truncated*-normal rates, which is not
    identical to ``lba2`` (see ``tests/test_mtr.py``).
"""

from ssms.basic_simulators import boundary_functions as bf
from ssms.basic_simulators.mtr import POSSIBLE_CHOICES, multiple_threshold_race

MTR_PARAMS = ["A", "B0", "B1", "D0", "D1", "v0", "v1", "sv0", "sv1", "t"]
MTR_LOWER = [0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0]
MTR_UPPER = [3.0, 5.0, 5.0, 1.0, 1.0, 6.0, 6.0, 3.0, 3.0, 2.0]
# Defaults in the paper's units, close to its Experiment 1 high-error-cost,
# accuracy-emphasis estimates (Figs 7-8; A, sv-match and t0 are not printed).
MTR_DEFAULTS = [1.0, 2.9, 3.05, 0.78, 0.88, 3.55, 2.99, 1.0, 1.0, 0.2]


def get_mtr_config() -> dict:
    """Return the ``mtr`` model configuration."""
    return {
        "name": "mtr",
        "params": list(MTR_PARAMS),
        "param_bounds": [list(MTR_LOWER), list(MTR_UPPER)],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": len(MTR_PARAMS),
        "default_params": list(MTR_DEFAULTS),
        "nchoices": 3,
        "choices": list(POSSIBLE_CHOICES),
        "n_particles": 2,
        "tags": ["python_simulator"],
        "simulator": multiple_threshold_race,
        "parameter_transforms": {"sampling": [], "simulation": []},
    }
