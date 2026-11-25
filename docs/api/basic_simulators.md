# Basic simulators API

The root `ssms.basic_simulators` namespace exports `Simulator`, the legacy
`simulator` module, `boundary_functions`, `drift_functions`,
`modular_parameter_simulator_adapter`, `normalize_simulator_result`, and
`validate_observation_result`.

## Observation constants

<span id="ssms.OMISSION_SENTINEL" aria-hidden="true"></span>
<span id="ssms.basic_simulators.OBSERVATION_SCHEMA_VERSION" aria-hidden="true"></span>
<span id="ssms.basic_simulators.OMISSION_SENTINEL" aria-hidden="true"></span>
<span id="ssms.basic_simulators.observation_results.OBSERVATION_SCHEMA_VERSION" aria-hidden="true"></span>

| Export | Value and role |
| --- | --- |
| `OBSERVATION_SCHEMA_VERSION` | Current native structured-observation schema version (`1`) |
| `OMISSION_SENTINEL` | Package-wide omitted/deadline response marker (`-999.0`) |

The package-native structured-observation validator is exported from
`ssms.basic_simulators`. See the [contract reference](../reference/observation_result_contract.md)
for its versioned schema and compatibility boundary. Existing `rts`/`choices` consumers
can opt into the canonical view with the
[legacy-result normalization guide](../how_to/normalize_simulator_results.md).

For HSSM's classic simulator wrapper contract, see the
[HSSM integration API](hssm_support.md). Parameter preparation is documented in
the [parameter-transform API](parameter_adapters.md).

## Fixed circular diffusion

`circular_diffusion` simulates the first exit of a centered planar diffusion
with constant Cartesian drift `vx, vy` from a circle of radius `a`. Total RT
includes the nondecision time `t`. Within-trial noise is isotropic (unit scale
by default). Unlike `angle`, which names a collapsing-boundary DDM, its
response is a **continuous direction** in radians on `[-π, π)`.

```python
from ssms.basic_simulators import Simulator

result = Simulator(model="circular_diffusion").simulate(
    theta={"vx": 0.8, "vy": 0.6, "a": 1.0, "t": 0.2},
    n_samples=200, random_state=42,
)
rt, direction = result["observations"][:, 0].T
```

The legacy `choices` slot also contains the **floating-point angle**, not a
category. Native omissions are all-NaN rows with a boolean `omission_mask`;
the legacy slots use `-999`. Decision times exceeding `max_t` or total RTs
reaching a deadline are omissions. Categorical probabilities, choice-binned
histograms, and the built-in categorical LAN pipeline are not applicable.

This continuous-time sampler uses the independent exit-angle and exit-time laws
in [Smith (2016)](https://doi.org/10.1037/rev0000023). It samples the von Mises
angle directly. For decision time it samples 128 Bessel modes and replaces the
remainder by its mean (RMS truncation error below 0.000082 times radius squared
divided by noise variance). `delta_t` and `smooth_unif` are not used.

::: ssms.basic_simulators
