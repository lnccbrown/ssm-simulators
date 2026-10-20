# Parameter-transform API

`ParameterTransform` is the canonical abstract interface for sampling- and
simulation-time parameter transforms. `ParameterAdaptation` is its
backward-compatible public alias under
`ssms.basic_simulators.parameter_adapters`.

Use [Write a custom parameter transform](../contributing/add_parameter_adapters.md)
for the implementation and testing workflow.

::: ssms.transforms.base.ParameterTransform

## Adapter registry

The compatibility namespace also exports `ParameterAdapterRegistry`,
`register_adapter_to_model`, `register_adapter_to_model_family`, and
`get_adapter_registry`. Their generated reference remains on the
[basic simulators API](basic_simulators.md); this page owns the canonical
transform interface and alias relationship without rendering those objects a
second time.

## Built-in MLBA front end

`MLBAFrontEnd` is the simulation-time transform behind the `mlba2` and `mlba3`
models (Trueblood, Brown & Heathcote, 2014). It maps per-option attribute
values onto LBA mean drift rates and is exported from both
`ssms.transforms.simulation` and `ssms.transforms`. The pure-NumPy equations it
evaluates live in `ssms.transforms.simulation.mlba_front_end`.

::: ssms.transforms.simulation.mlba.MLBAFrontEnd

::: ssms.transforms.simulation.mlba_front_end
