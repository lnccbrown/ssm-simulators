import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Quick Start
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `ssms` package serves two purposes.

    1. Easy access to *fast simulators of sequential sampling models*

    2. Support infrastructure to construct training data for various approaches to likelihood / posterior amortization

    We provide two minimal examples here to illustrate how to use each of the two capabilities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Install

    Let's start with *installing* the `ssms` package.

    You can do so by typing,

    `pip install ssm-simulators`

    in your terminal.

    Below you find a basic tutorial on how to use the package.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Tutorial
    """)
    return


@app.cell
def _():
    # Import necessary packages
    import numpy as np
    import pandas as pd
    import ssms
    return (ssms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Using the Simulators

    Let's start with using the basic simulators.
    You access the main simulators through the  `ssms.basic_simulators.simulator.simulator()` function.

    To get an idea about the models included in `ssms`, use the `config` module.
    The central dictionary with metadata about included models sits in `ssms.config.model_config`.
    """)
    return


@app.cell
def _(ssms):
    # Check included models
    list(ssms.config.model_config.keys())[:10]
    return


@app.cell
def _(ssms):
    # Take an example config for a given model
    ssms.config.model_config["ddm"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note:**
    The usual structure of these models includes,

    - Parameter names (`'params'`)
    - Bounds on the parameters (`'param_bounds'`)
    - A function that defines a boundary for the respective model (`'boundary'`)
    - The number of parameters (`'n_params'`)
    - Defaults for the parameters (`'default_params'`)
    - The number of choices the process can produce (`'nchoices'`)

    The `'hddm_include'` key concerns information useful for integration with the [hddm](https://github.com/hddm-devs/hddm) python package, which facilitates hierarchical bayesian inference for sequential sampling models. It is not important for the present tutorial.
    """)
    return


@app.cell
def _():
    from ssms.basic_simulators.simulator import simulator

    sim_out = simulator(
        model="ddm", theta={"v": 0, "a": 1, "z": 0.5, "t": 0.5}, n_samples=1000,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output of the simulator is a `dictionary` with three elements.

    1. `rts` (array)
    2. `choices` (array)
    3. `metadata` (dictionary)

    The `metadata` includes the named parameters, simulator settings, and more.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Using the Training Data Generators

    The training data generators sit on top of the simulator function to turn raw simulations into usable training data for training machine learning algorithms aimed at posterior or likelihood armortization.

    We will use the `data_generator` class from `ssms.dataset_generators`. Initializing the `data_generator` boils down to supplying two configuration dictionaries.

    1. The `generator_config`, concerns choices as to what kind of training data one wants to generate.
    2. The `model_config` concerns choices with respect to the underlying generative *sequential sampling model*.

    We will consider a basic example here, concerning data generation to prepare for training [LANs](https://elifesciences.org/articles/65074).

    Let's start by peeking at an example `generator_config`.
    """)
    return


@app.cell
def _(ssms):
    ssms.config.data_generator_config["lan"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You usually have to make just few changes to this basic configuration dictionary.
    An example below.
    """)
    return


@app.cell
def _(ssms):
    from copy import deepcopy

    # Initialize the generator config (for MLP LANs)
    generator_config = deepcopy(ssms.config.data_generator_config["lan"])
    # Specify generative model (one from the list of included models mentioned above)
    generator_config["dgp_list"] = "angle"
    # Specify number of parameter sets to simulate
    generator_config["n_parameter_sets"] = 100
    # Specify how many samples a simulation run should entail
    generator_config["n_samples"] = 1000
    return (generator_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's define our corresponding `model_config`.
    """)
    return


@app.cell
def _(ssms):
    model_config = ssms.config.model_config["angle"]
    print(model_config)
    return (model_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are now ready to initialize a `data_generator`, after which we can generate training data using the `generate_data_training_uniform` function, which will use the hypercube defined by our parameter bounds from the `model_config` to uniformly generate parameter sets and corresponding simulated datasets.
    """)
    return


@app.cell
def _(generator_config, model_config, ssms):
    my_dataset_generator = ssms.dataset_generators.lan_mlp.data_generator(
        generator_config=generator_config, model_config=model_config
    )
    return (my_dataset_generator,)


@app.cell
def _(my_dataset_generator):
    training_data = my_dataset_generator.generate_data_training_uniform(save=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `training_data` is a dictionary containing four keys:

    1. `data` the features for [LANs](https://elifesciences.org/articles/65074), containing vectors of *model parameters*, as well as *rts* and *choices*.
    2. `labels` which contain approximate likelihood values
    3. `generator_config`, as defined above
    4. `model_config`, as defined above
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can now use this training data for your purposes. If you want to train [LANs](https://elifesciences.org/articles/65074) yourself, you might find the [LANfactory](https://github.com/AlexanderFengler/LANfactory) package helpful.

    You may also simply find the basic simulators provided with the **ssms** package useful, without any desire to use the outputs into training data for amortization purposes.

    ##### END
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
