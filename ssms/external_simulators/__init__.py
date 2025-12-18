"""External simulator interfaces for ssms.

This module provides wrappers and mappers for integrating external
simulation and modeling packages with ssms.

SSMSToPyDDMMapper
-----------------
Converts ssms model configurations to PyDDM models for analytical
Fokker-Planck solutions.

Tested compatibility:
- DDM and variants (ddm, ddm_par2, ddm_seq2, ddm_mic2_*)
- Ornstein models (position-dependent drift)
- Collapsing boundary models (angle, weibull)
- Custom drift models (gamma_drift, conflict_*, attend_*, shrink_spot_*)

Features:
- Handles parameter conflicts ('t' non-decision time vs time variable)
- Returns array-compatible drift/boundary functions
- Automatic z transformation from [0,1] to [-a, a]
- Uses modern PyDDM API (solution.pdf("correct"))

Incompatible models:
- Multi-particle: race_*, lca_*, lba*
- Non-Gaussian noise: levy
- Inter-trial variability: full_ddm, ddm_sdv, ddm_st
"""

from ssms.external_simulators.pyddm_mapper import SSMSToPyDDMMapper

__all__ = ["SSMSToPyDDMMapper"]
