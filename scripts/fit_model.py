#!/usr/bin/env python
import json

import argparse
import numpy as np
from tcup import tcup

# Set up run parameters
SEED = 24601
N_SAMPLES = 5000


def load_dataset(filename):
    with open(filename, "r") as f:
        dataset = json.load(f)

    data = {key: np.array(val) for key, val in dataset["data"].items()}
    params = dataset["info"]
    return data, params


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fit a model to data")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-f", "--fixed", type=int)
    group.add_argument("-n", "--normal", action="store_true")
    parser.add_argument("dataset")
    parser.add_argument("outfile")
    args = parser.parse_args()

    # Load dataset
    data, params = load_dataset(args.dataset)

    # Fit model
    if args.normal:
        mcmc = tcup(data, SEED, model="ncup", num_samples=N_SAMPLES)
    elif args.fixed:
        data["nu"] = args.fixed
        mcmc = tcup(data, SEED, num_samples=N_SAMPLES)
    else:
        mcmc = tcup(data, SEED, num_samples=N_SAMPLES)

    # Save chains
    mcmc.to_netcdf(args.outfile)
