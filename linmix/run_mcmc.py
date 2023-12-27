#!/usr/bin/env python2
# NB this script is in Python 2 to work with linmix
import argparse
import json

import numpy as np
import linmix


def load_dataset(filename):
    with open(filename, "r") as f:
        dataset = json.load(f)

    data = {key: np.array(val) for key, val in dataset["data"].items()}
    params = dataset.get("info")
    return data, params


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fit linmix to data")
    parser.add_argument("-r", "--random-seed", type=int)
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-K", "--num-components", type=int, default=2)
    parser.add_argument("dataset")
    parser.add_argument("outfile")
    args = parser.parse_args()

    # Load dataset
    data, params = load_dataset(args.dataset)

    # Set seed
    np.random.seed(args.random_seed)

    # Fit model
    lm = linmix.LinMix(
        x=data["x"],
        y=data["y"],
        xsig=data["dx"],
        ysig=data["dy"],
        K=args.num_components,
    )
    lm.run_mcmc(silent=args.quiet)

    # Convert LinMix object to dict
    linmix_keys = [
        "alpha",
        "beta",
        "sigsqr",
        "pi",
        "mu",
        "tausqr",
        "mu0",
        "usqr",
        "wsqr",
        "ximean",
        "xisig",
        "corr",
    ]
    mcmc = {key: lm.chain[:][key].tolist() for key in linmix_keys}

    # Save output
    with open(args.outfile, "w") as f:
        json.dump(mcmc, f)
