import numpy as np
import pandas as pd
import ssms
import argparse
import logging
import os, sys
from time import time
from pprint import pformat
import matplotlib.pyplot as plt

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

def main():
    parser = argparse.ArgumentParser(
        prog="test model",
        description="tests the given model by generating data from it"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="ds_conflict_stimflexons_leak_decay",
        help="Name of the model to use for modelconfig.\ne.g. ds_conflict_stimflexons(_leak)_drift(_angle)"
    )
    # parser.add_argument(
    #     "--genmodel", "-g", type=str, required=False,
    #     help="Also model name, but now for the generator config (same as --model if not provided)."
    # )
    parser.add_argument(
        "--name", "-n", type=str, default="",
        help="Name to save the figure as!"
    )
    parser.add_argument(
        "--save", "-v", default=False, action=argparse.BooleanOptionalAction,
        help="Save to disk?"
    )
    parser.add_argument(
        "--samples", "-s", type=int, default=1000,
        help="n_samples"
    )
    parser.add_argument(
        "--params", "-p", type=int, default=1000,
        help="n_parameter_sets"
    )
    parser.add_argument(
        "--boundseparation", "-a", type=float, default=1,
        help="boundary separation"
    )
    parser.add_argument(
        "--decay", "-g", type=float, default=0,
        help="decay rate"
    )
    parser.add_argument(
        "--tonset", "-t", type=float, default=1,
        help="target onset time (how long the stimulus is present for?)"
    )
    parser.add_argument(
        "--donset", "-d", type=float, default=0.5,
        help="distractor onset time (how long the stimulus is present for?)"
    )
    parser.add_argument(
        "--tcoh", "-c", type=float, default=1,
        help="target coherence"
    )
    parser.add_argument(
        "--dcoh", "-C", type=float, default=0,
        help="distractor coherence"
    )
    parser.add_argument(
        "--starting", "-z", type=float, default=0.5,
        help="vertical starting point between the boundaries (0 to 1)"
    )
    parser.add_argument(
        "--nondecision", "-T", type=float, default=0,
        help="non-decision time (how long before the stimulus starts acting)"
    )
    parser.add_argument(
        "--tfixedp", "-f", type=float, default=0.001,
        help="t fixed p (???)"
    )
    parser.add_argument(
        "--timelimit", "-l", type=int, default=2000,
        help="time limit for trajectory graph"
    )
    parser.add_argument(
        "--timestart", "-L", type=int, default=250,
        help="time start for trajectory graph"
    )
    parser.add_argument(
        "--figwidth", "-w", type=int, default=200,
        help="width for trajectory graph"
    )
    parser.add_argument(
        "--figheight", "-W", type=int, default=50,
        help="height for trajectory graph"
    )

    log_config()

    args = parser.parse_args()
    # genmodel = args.model #if args.genmodel is None else args.genmodel
    
    '''
    modfig = ssms.config.model_config[args.model]
    # logging.info(f"got modfig")
    genfig = ssms.config.get_default_generator_config("lan")
    # logging.info(f"got genfig")
    genfig["model"] = genmodel
    genfig["n_samples"] = args.samples
    genfig["n_parameter_sets"] = args.params
    logging.info(f"modfig = " + pformat(modfig) + "\n\n")
    logging.info(f"genfig = " + pformat(genfig) + "\n\n")

    datagen = ssms.dataset_generators.lan_mlp.data_generator(generator_config = genfig, model_config=modfig)
    logging.info(f"got datagen")
    # logging.info(f"{datagen.get_simulations()}\n\n")
    data = datagen.generate_data_training_uniform(save=args.save)
    logging.info(f"generated data!")
    logging.info(f"{list(data.keys()) = }\n")
    logging.info(f"{data['cpn_data'] = }")
    return data
    '''

    leak_theta = {
            "a": args.boundseparation,
            "z" : args.starting,
            "g" : args.decay,
            "t" : args.nondecision,
            "tinit" : 0,
            "dinit" : 0,
            "tslope" : 0,
            "dslope" : 0,
            "tfixedp" : args.tfixedp,
            "tcoh" : args.tcoh,
            "dcoh" : args.dcoh,
            "tonset" : args.tonset,
            "donset" : args.donset,
        }
    norm_theta = {
            "a": args.boundseparation,
            "z": args.starting,
            "t": args.nondecision,
            "tinit": 0,
            "dinit": 0,
            "tslope": 0,
            "dslope": 0,
            "tfixedp": args.tfixedp,
            "tcoh": args.tcoh,
            "dcoh": args.dcoh,
            "tonset": args.tonset,
            "donset": args.donset,
        }
    theta = leak_theta if "leak" in args.model else norm_theta
    out = ssms.basic_simulators.simulator.simulator(
        model=args.model, theta=theta, n_samples=args.samples
    )
    print(f"choices: 1 -> {np.count_nonzero(out['choices'] == 1)}; -1 -> {np.count_nonzero(out['choices'] == -1)}; total -> {len(out['choices'])}")
    traj = out["metadata"]["trajectory"][args.timestart:args.timelimit]

    plt.figure(figsize=(args.figwidth,args.figheight))
    plt.plot(np.arange(len(traj)), traj, linewidth=6)
    plt.savefig((f"data/{args.name},a-{args.boundseparation},z-{args.starting},g-{args.decay},"
    f"T-{args.nondecision},f-{args.tfixedp},t-{args.tonset},d-{args.donset},"
    f"c-{args.tcoh},C-{args.dcoh},l-{args.timelimit},L-{args.timestart};{round(time())%100000}.png"))


if __name__ == "__main__":
    main()