# Simulator and data-generation capabilities

`ssm-simulators` owns two related capabilities in the HSSM ecosystem: fast
forward simulation from sequential-sampling models and generation of labeled
datasets for likelihood-approximation networks. The first produces synthetic
behavior; the second repeatedly uses simulation or an analytical estimator to
construct training examples.

## Direct simulation

The simulator layer covers classic diffusion models, collapsing-boundary and
time-varying-drift variants, multi-choice accumulator models, attention models,
and reinforcement-learning SSMs. Most users select a registered model and call
the class-based `Simulator` interface. Model configurations describe parameters,
bounds, choices, boundary/drift components, and parameter transforms; they do
not contain a participant's concrete parameter values.

Use the [basic tutorial](../basic_tutorial/basic_tutorial.ipynb) for a first
completed simulation. The [configuration guide](../core_tutorials/tutorial_configs.ipynb)
explains how to select and adapt registered models, while the
[custom-model guide](../core_tutorials/tutorial_custom_models.ipynb) covers
runtime extensions.

## Training-data generation

The data-generation layer samples valid parameter combinations, obtains
likelihood labels through simulation-based KDE or compatible PyDDM solutions,
and emits the feature/label arrays consumed by LANfactory. It also supports
specialized targets such as choice- and omission-probability labels. LANfactory,
not this package, owns network architecture and training behavior.

Use [Generate training data for LANs](../core_tutorials/tutorial_data_generators.ipynb)
for the task workflow. The [data-generator API](../api/dataset_generators.md)
defines the programmatic surface.

## Choosing a likelihood path

Simulation works across the broadest model set and can approximate a likelihood
with KDE. PyDDM instead solves the Fokker–Planck equation for compatible
single-particle, two-choice, Gaussian-noise models. The
[simulation and PyDDM explanation](../core_tutorials/tutorial_simulators_vs_pyddm.ipynb)
describes the trade-off and compatibility boundary; the
[KDE guide](../core_tutorials/kde_class.ipynb) covers the simulation-based path.

## Ecosystem boundary

`ssm-simulators` owns simulator definitions, model metadata, synthetic data,
and training-data construction. LANfactory owns training and exporting neural
likelihoods, and HSSM owns Bayesian inference. See the
[HSSM ecosystem map](https://lnccbrown.github.io/HSSM/ecosystem/) for the full
artifact flow.
