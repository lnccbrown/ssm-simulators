# Data-generator API

The data-generator surface assembles parameter sampling, simulation or
analytical likelihood estimation, and labeled training examples. Use
[Generate training data for LANs](../core_tutorials/tutorial_data_generators.ipynb)
for the procedure.

## Root namespace

::: ssms.dataset_generators

## Mixture strategy

`MixtureTrainingStrategy` is the backward-compatible public alias of
`ResampleMixtureStrategy`. New code should use the canonical class name; both
names construct the same implementation.

::: ssms.dataset_generators.strategies.ResampleMixtureStrategy
