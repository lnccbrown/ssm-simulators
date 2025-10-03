import numpy as np
import pandas as pd
import ssms
import argparse
import logging
import os, sys
from time import time
from pprint import pformat

"""
    Usage:
    python test.py -m <MODELNAME> ... 
"""

def log_config(filename=f".log/{time()}.txt"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logging.basicConfig(
        # filename = filename,
        handlers = [
            logging.FileHandler(filename),
            logging.StreamHandler(sys.stdout)
        ],
        level = logging.INFO
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="test model",
        description="tests the given model by generating data from it"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="ds_conflict_stimflexons_drift",
        help="Name of the model to use for modelconfig.\ne.g. ds_conflict_stimflexons(_leak)_drift(_angle)"
    )
    parser.add_argument(
        "--genmodel", "-g", type=str, required=False,
        help="Also model name, but now for the generator config (same as --model if not provided)."
    )
    parser.add_argument(
        "--save", "-v", default=False, action=argparse.BooleanOptionalAction,
        help="Save to disk?"
    )
    parser.add_argument(
        "--samples", "-s", type=int, default=200,
        help="n_samples"
    )
    parser.add_argument(
        "--params", "-p", type=int, default=200,
        help="n_parameter_sets"
    )

    log_config()

    args = parser.parse_args()
    genmodel = args.model if args.genmodel is None else args.genmodel
    
    modfig = ssms.config.model_config[args.model]
    logging.info(f"got modfig")
    genfig = ssms.config.get_default_generator_config("lan")
    logging.info(f"got genfig")
    genfig["model"] = genmodel
    genfig["n_samples"] = args.samples
    genfig["n_parameter_sets"] = args.params
    logging.info(f"modfig = " + pformat(modfig) + "\n\n")
    logging.info(f"genfig = " + pformat(genfig) + "\n\n")

    datagen = ssms.dataset_generators.lan_mlp.data_generator(generator_config = genfig, model_config=modfig)
    logging.info(f"got datagen")
    logging.info(f"{dir(datagen)=}\n\n")
    data = datagen.generate_data_training_uniform(save=args.save)
    logging.info(f"generated data!")
    logging.info(f"{list(data.keys()) = }\n")
    logging.info(f"{data['cpn_data'] = }")
