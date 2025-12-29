# ssm_simulators
Python Package which collects simulators for Sequential Sampling Models.

### Quick Start

The `ssms` package serves two purposes.

1. Easy access to *fast simulators of sequential sampling models*

2. Support infrastructure to construct training data for various approaches to likelihood / posterior amortization

We provide two minimal examples here to illustrate how to use each of the two capabilities.

#### Install

Let's start with *installing* the `ssms` package.

You can do so by typing,

`pip install git+https://github.com/AlexanderFengler/ssm_simulators`

in your terminal.

Below you find a basic tutorial on how to use the package.

#### Tutorial

```python
# Import necessary packages
import numpy as np
import pandas as pd
import ssms
```

#### Using the Simulators

Let's start with using the basic simulators.
You access the main simulators through the  `ssms.basic_simulators.simulator` function.

To get an idea about the models included in `ssms`, use the `config` module.
The central dictionary with metadata about included models sits in `ssms.config.model_config`.


```python
# Check included models
list(ssms.config.model_config.keys())[:10]
```




    ['ddm',
     'ddm_legacy',
     'angle',
     'weibull',
     'levy',
     'levy_angle',
     'full_ddm',
     'ornstein',
     'ornstein_angle',
     'ddm_sdv']




```python
# Take an example config for a given model
ssms.config.model_config['ddm']
```




    {'name': 'ddm',
     'params': ['v', 'a', 'z', 't'],
     'param_bounds': [[-3.0, 0.3, 0.1, 0.0], [3.0, 2.5, 0.9, 2.0]],
     'boundary': <function ssms.basic_simulators.boundary_functions.constant(t=0)>,
     'n_params': 4,
     'default_params': [0.0, 1.0, 0.5, 0.001],
     'hddm_include': ['z'],
     'nchoices': 2}



**Note:**
The usual structure of these models includes,

- Parameter names (`'params'`)
- Bounds on the parameters (`'param_bounds'`)
- A function that defines a boundary for the respective model (`'boundary'`)
- The number of parameters (`'n_params'`)
- Defaults for the parameters (`'default_params'`)
- The number of choices the process can produce (`'nchoices'`)

The `'hddm_include'` key concerns information useful for integration with the [hddm](https://github.com/hddm-devs/hddm) python package, which facilitates hierarchical bayesian inference for sequential sampling models. It is not important for the present tutorial.


```python
from ssms.basic_simulators import simulator
sim_out = simulator(model = 'ddm',
                    theta = [0, 1, 0.5, 0.5],
                    n_samples = 1000)
```

The output of the simulator is a `dictionary` with three elements.

1. `rts` (array)
2. `choices` (array)
3. `metadata` (dictionary)

The `metadata` includes the named parameters, simulator settings, and more.

#### Using the Training Data Generators

The training data generators sit on top of the simulator function to turn raw simulations into usable training data for training machine learning algorithms aimed at posterior or likelihood armortization.

We will use the `DataGenerator` class from `ssms.dataset_generators`. Initializing the `DataGenerator` boils down to supplying two configuration dictionaries.

1. The `generator_config`, concerns choices as to what kind of training data one wants to generate.
2. The `model_config` concerns choices with respect to the underlying generative *sequential sampling model*.

We will consider a basic example here, concerning data generation to prepare for training [LANs](https://elifesciences.org/articles/65074).

Let's start by peeking at an example `generator_config`.


```python
ssms.config.DataGenerator_config['lan']['mlp']
```




    {'output_folder': 'data/lan_mlp/',
     'dgp_list': 'ddm',
     'nbins': 0,
     'n_samples': 100000,
     'n_parameter_sets': 10000,
     'n_parameter_sets_rejected': 100,
     'n_training_samples_by_parameter_set': 1000,
     'max_t': 20.0,
     'delta_t': 0.001,
     'pickleprotocol': 4,
     'n_cpus': 'all',
     'kde_data_mixture_probabilities': [0.8, 0.1, 0.1],
     'simulation_filters': {'mode': 20,
      'choice_cnt': 0,
      'mean_rt': 17,
      'std': 0,
      'mode_cnt_rel': 0.9},
     'negative_rt_cutoff': -66.77497,
     'n_subruns': 10,
     'bin_pointwise': False,
     'separate_response_channels': False}



You usually have to make just few changes to this basic configuration dictionary.
An example below.


```python
from copy import deepcopy
# Initialize the generator config (for MLP LANs)
generator_config = deepcopy(ssms.config.DataGenerator_config['lan']['mlp'])
# Specify generative model (one from the list of included models mentioned above)
generator_config['dgp_list'] = 'angle'
# Specify number of parameter sets to simulate
generator_config['n_parameter_sets'] = 100
# Specify how many samples a simulation run should entail
generator_config['n_samples'] = 1000
```

Now let's define our corresponding `model_config`.


```python
model_config = ssms.config.model_config['angle']
print(model_config)
```

    {'name': 'angle', 'params': ['v', 'a', 'z', 't', 'theta'], 'param_bounds': [[-3.0, 0.3, 0.1, 0.001, -0.1], [3.0, 3.0, 0.9, 2.0, 1.3]], 'boundary': <function angle at 0x11b2a7c10>, 'n_params': 5, 'default_params': [0.0, 1.0, 0.5, 0.001, 0.0], 'hddm_include': ['z', 'theta'], 'nchoices': 2}


We are now ready to initialize a `DataGenerator`, after which we can generate training data using the `generate_data_training_uniform` function, which will use the hypercube defined by our parameter bounds from the `model_config` to uniformly generate parameter sets and corresponding simulated datasets.


```python
my_dataset_generator = ssms.dataset_generators.DataGenerator(generator_config = generator_config,
                                                              model_config = model_config)
```

    n_cpus used:  6
    checking:  data/lan_mlp/



```python
training_data = my_dataset_generator.generate_data_training_uniform(save = False)
```

    simulation round: 1  of 10
    simulation round: 2  of 10
    simulation round: 3  of 10
    simulation round: 4  of 10
    simulation round: 5  of 10
    simulation round: 6  of 10
    simulation round: 7  of 10
    simulation round: 8  of 10
    simulation round: 9  of 10
    simulation round: 10  of 10


`training_data` is a dictionary containing four keys:

1. `data` the features for [LANs](https://elifesciences.org/articles/65074), containing vectors of *model parameters*, as well as *rts* and *choices*.
2. `labels` which contain approximate likelihood values
3. `generator_config`, as defined above
4. `model_config`, as defined above

You can now use this training data for your purposes. If you want to train [LANs](https://elifesciences.org/articles/65074) yourself, you might find the [LANfactory](https://github.com/AlexanderFengler/LANfactory) package helpful.

You may also simply find the basic simulators provided with the **ssms** package useful, without any desire to use the outputs into training data for amortization purposes.

##### END
