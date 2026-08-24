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

::: ssms.basic_simulators
