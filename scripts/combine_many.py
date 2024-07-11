import argparse
import warnings

import arviz as az
import numpy as np
import pandas as pd
import scipy.stats as sps
from tqdm import trange
import xarray as xr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("-n", "--num-repeats", default=1000, type=int)
    args = parser.parse_args()

    var_names = [
        "alpha",
        "beta",
    ]
    if args.model == "ncup":
        var_names += ["sigma"]
    else:
        var_names += ["sigma_68"]
    if args.model in ["tcup", "tobs"]:
        var_names += ["nu"]

    results = None

    for idx in trange(args.num_repeats):
        try:
            mcmc = az.from_netcdf(
                f"results/fixed/{args.model}/{args.dataset}/{idx+1}.nc"
            )
        except FileNotFoundError:
            warnings.warn("File not found")
            continue

        if results is None:
            results = az.extract(mcmc, var_names=var_names, num_samples=1000)
        else:
            next_results = az.extract(
                mcmc, var_names=var_names, num_samples=1000
            )
            results = xr.concat([results, next_results], dim="sample")

    if args.model == "ncup":
        results = results.rename({"sigma": "sigma_68"})

    results = results.reset_index(["sample", "chain", "draw"], drop=True)
    results.to_netcdf(f"results/{args.dataset}_{args.model}_many_samples.nc")
