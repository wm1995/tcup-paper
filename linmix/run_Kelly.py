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
    # Load dataset
    data, params = load_dataset("kelly.json")

    # Set seed
    np.random.seed(0)

    # Fit model
    lm = linmix.LinMix(
        x=data["x"],
        y=data["y"],
        xsig=data["dx"],
        ysig=data["dy"],
        K=2,
    )

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
    with open("linmix-kelly.json", "w") as f:
        json.dump(mcmc, f)
