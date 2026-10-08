# HSSM integration API

`ssms.hssm_support` adapts classic ssms simulator functions to HSSM's random
variable interface. It decorates simulator callables with `model_name`,
`choices`, and `obs_dim`, validates those attributes, broadcasts parameter and
trial covariate shapes, and returns `(rt, response)` observations in HSSM's
expected layout.

This module is the classic SSM bridge. RLSSM inference uses the separate
[assembled-model contract](rlssm.md#assembled-model-and-hssm-contracts).

::: ssms.hssm_support
