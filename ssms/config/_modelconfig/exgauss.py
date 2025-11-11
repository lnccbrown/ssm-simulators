"""Ex Gauss model configuration."""

import cssm
from ssms.basic_simulators import boundary_functions as bf

def get_exgauss_config(): 
    """Get the configuration of the Ex-Gaussian model""" 
    return { 
        "name": "exgauss", 
        "params": ["mu0", "mu1", "sigma0", "sigma1", "tau0", "tau1", "p"], 
        "param_bounds": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                         [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 1.0]], 
        "boundary_name": 'constant', 
        "boundary": bf.constant, 
        "boundary_params": [], 
        "n_params": 7, 
        "default_params": [0.5, 0.5, 0.05, 0.05, 0.3, 0.3, 0.5], 
        "nchoices": 2, 
        "choices": [-1, 1], 
        "n_particles": 1, 
        "simulator": cssm.exgauss, 
    }

def get_exgauss_race_config(): 
    """Get the configuration of the race version of Ex-Gaussian model"""
    return { 
        "name": "exgauss_race",
        "params": ["mu0", "mu1", "sigma0", "sigma1", "tau0", "tau1", "p"], 
        "param_bounds": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                         [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 1.0]],
        "boundary_name": 'constant', 
        "boundary": bf.constant, 
        "boundary_params": [], 
        "n_params": 7, 
        "default_params": [0.5, 0.5, 0.05, 0.05, 0.3, 0.3, 0.5], 
        "nchoices": 2, 
        "choices": [-1, 1], 
        "n_particles": 1, 
        "simulator": cssm.exgauss_race,
    } 