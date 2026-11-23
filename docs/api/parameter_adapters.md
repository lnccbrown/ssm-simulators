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
