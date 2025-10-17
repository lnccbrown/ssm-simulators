"""DDM model configuration."""

import cssm
from ssms.basic_simulators import boundary_functions as bf

def get_exgauss_config(): 
    """Get the configuration of the Ex-Gaussian model""" 
    return { 
        "name": "exgauss", 
        "params": ["mu", "sigma", "tau"], 
        "param_bounds": [[-50.0, 0.0, 0.0], [50.0, 50.0, 50.0]], 
        "boundary_name": None, 
        "boundary": None, 
        "boundary_params": [], 
        "n_params": 3, 
        "default_params": [0.5, 0.05, 0.3], 
        "nchoices": 1, 
        "choices": [None], 
        "n_particles": 1, 
        "simulator": cssm.exgauss, 
    }