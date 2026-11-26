"""Sensitivity of the reproduced quantities to the unprinted A and sv_c.

Recomputes the DK slope and the mean p(don't know) per error-cost group at the
corners of A in {0.5, 2.0}, sv_c in {0.7, 1.4} (bridge defaults: 1.0, 1.0).
Writes sensitivity.txt. Run from the repository root:
    MTR_BUNDLE_DIR=<bundle>/contribution .venv/bin/python -m tests.mtr_audits.sensitivity
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# ``bridge.py``/``reproduce.py`` are checker-only artifacts of the generated
# contribution bundle and are not part of the library; point MTR_BUNDLE_DIR at
# the bundle's ``contribution/`` directory to run this audit.
_BUNDLE = os.environ.get("MTR_BUNDLE_DIR")
if not _BUNDLE:
    raise SystemExit("Set MTR_BUNDLE_DIR to the bundle's contribution/ directory.")
sys.path.insert(0, _BUNDLE)
import bridge
import reproduce

lines = []
base = dict(bridge.PAPER_PARAMS)
for A in (0.5, 1.0, 2.0):
    for svc in (0.7, 1.0, 1.4):
        if (A == 1.0) != (svc == 1.0):
            continue
        bridge.PAPER_PARAMS.update(A=A, sv_c=svc)
        lo, hi = reproduce._mean_pdk("low"), reproduce._mean_pdk("high")
        s_lo, s_hi = reproduce._dk_slope("low"), reproduce._dk_slope("high")
        line = (
            f"A={A} sv_c={svc}: slope low={s_lo:.3f} high={s_hi:.3f} | "
            f"p(DK) low={lo:.3f} high={hi:.3f} increase={hi - lo:.3f}"
        )
        print(line, flush=True)
        lines.append(line)
bridge.PAPER_PARAMS.clear()
bridge.PAPER_PARAMS.update(base)
with open(os.path.join(HERE, "sensitivity.txt"), "w") as f:
    f.write(
        "paper: slope low 2.02, high 1.84; fitted p(DK) low ~0.335, high ~0.425, increase ~0.08\n"
    )
    f.write("\n".join(lines) + "\n")
