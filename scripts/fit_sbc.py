#!/usr/bin/env python
import argparse
import arviz as az
import numpy as np
import stan
from tcup.stan import _get_model_src
from tcup_paper.data.io import load_dataset

MAX_SAMPLES = 200000
SAFETY_MARGIN = 1.1
PARAMS_OF_INTEREST = ["alpha_scaled", "beta_scaled", "sigma_scaled", "nu"]

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run a single chain for simulation-based calibration checks"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-f", "--fixed", type=float)
    group.add_argument("-n", "--normal", action="store_true")
    parser.add_argument("dataset")
    parser.add_argument("outfile")
    args = parser.parse_args()

    # Load dataset
    data, params = load_dataset(args.dataset)

    # Fit model
    if args.normal:
        model_src = _get_model_src("ncup")
        PARAMS_OF_INTEREST.remove("nu")
    elif args.fixed:
        model_src = _get_model_src("fixed")
        data["nu"] = args.fixed
        PARAMS_OF_INTEREST.remove("nu")
    else:
        # model_src = _get_model_src("tcup", nu_prior="pareto(1, 4)")
        model_src = _get_model_src(
            "tcup",
            reparam={
                "params": "real<lower=0> nu_less_one;",
                "transformed_params": "real nu = 1 + nu_less_one;",
                "half_nu": "real half_nu = nu / 2;",
                "prior": "nu_less_one ~ gamma(2, 1);",
            },
        )

    for seed in range(20):
        sampler = stan.build(model_src, data, random_seed=seed)
        n_samples = 5000

        for repeats in range(5):
            fit = sampler.sample(
                num_samples=n_samples,
                num_chains=1,
            )

            min_ess = (
                az.ess(fit, var_names=PARAMS_OF_INTEREST).to_array().min()
            )

            if min_ess > 1023:
                break
            else:
                n_samples = n_samples * (1023 / min_ess) * SAFETY_MARGIN
                n_samples = int(np.ceil(n_samples))
                if n_samples > MAX_SAMPLES:
                    # If the sampling will take too long, give up
                    break

        mcmc = az.from_pystan(fit)

        if min_ess > 1023:
            # Successful run - save output
            mcmc.to_netcdf(args.outfile)
            break
        else:
            # Chain length maxed out, save for diagnostic purposes
            filename = ".".join(args.outfile.split(".")[:-1])
            filename += f"_run_{sampler.random_seed}.nc"
            mcmc.to_netcdf(filename)
