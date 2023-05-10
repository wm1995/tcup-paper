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
    parser.add_argument("outfile_no_outlier")
    parser.add_argument("outfile_outlier")
    args = parser.parse_args()

    # This is only set up for the normal dataset
    assert args.dataset.split("/")[-1] == "normal.json"

    # Load dataset
    data, params = load_dataset(args.dataset)

    idx = params["outlier_idx"]
    mask = np.ones(shape=data["y"].shape, dtype=bool)
    mask[idx] = False
    no_outlier_data = {
        "x": data["x"][mask],
        "y": data["y"][mask],
        "dx": data["dx"][mask],
        "dy": data["dy"][mask],
    }

    # Fit model
    if args.normal:
        mcmc = tcup(data, SEED, model="ncup", num_samples=N_SAMPLES)
        no_outlier_mcmc = tcup(
            no_outlier_data, SEED, model="ncup", num_samples=N_SAMPLES
        )
    elif args.fixed:
        data["nu"] = args.fixed
        no_outlier_data["nu"] = args.fixed
        mcmc = tcup(data, SEED, num_samples=N_SAMPLES)
        no_outlier_mcmc = tcup(no_outlier_data, SEED, num_samples=N_SAMPLES)
    else:
        mcmc = tcup(data, SEED, num_samples=N_SAMPLES)
        no_outlier_mcmc = tcup(no_outlier_data, SEED, num_samples=N_SAMPLES)

    # Save chains
    mcmc.to_netcdf(args.outfile_outlier)
    no_outlier_mcmc.to_netcdf(args.outfile_no_outlier)
