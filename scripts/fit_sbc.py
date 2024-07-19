#!/usr/bin/env python
import argparse

import arviz as az
import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray
from tcup.numpyro import model_builder
from tcup_paper.data.io import load_dataset

MAX_SAMPLES = 200000
SAFETY_MARGIN = 1.1
PARAMS_OF_INTEREST = ["alpha_scaled", "beta_scaled", "sigma_68_scaled", "nu"]
L = 1023


def mixture_prior(weights, means, vars):
    if weights.shape == (1,):
        return dist.MultivariateNormal(
            loc=means[0],
            covariance_matrix=vars[0],
        )
    else:
        return dist.MixtureSameFamily(
            dist.CategoricalProbs(weights),
            dist.MultivariateNormal(
                loc=means,
                covariance_matrix=vars,
            ),
        )


def extend_inference_data(data, new_data):
    new_groups = {}
    # Loop over all InferenceData groups
    for group in data.groups():
        if group == "observed_data":
            # This is just the dataset provided to the MCMC sampler
            # So will be the same for both data and new_data
            new_groups[group] = data[group]
        else:
            # Concatenate group on draw
            new_groups[group] = xarray.concat(
                [data[group], new_data[group]],
                "draw",
            )
            # Reindex draw
            new_groups[group]["draw"] = np.arange(
                new_groups[group].sizes["draw"]
            )

    return az.InferenceData(**new_groups)


def build_post_pred_samples(results: az.InferenceData):
    az_samples = az.extract(results, num_samples=L, rng=0).to_dict()

    numpyro_samples = {}
    for var_name, var_samples in az_samples["data_vars"].items():
        data = np.array(var_samples["data"])
        data = np.moveaxis(data, -1, 0)
        numpyro_samples[var_name] = data

    return numpyro_samples


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run a single chain for simulation-based calibration checks"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-f", "--fixed", type=float)
    group.add_argument("-n", "--normal", action="store_true")
    parser.add_argument("-o", "--t-obs", action="store_false")
    parser.add_argument("dataset")
    parser.add_argument("outfile")
    args = parser.parse_args()

    # Load dataset
    data, params = load_dataset(args.dataset)

    rng_key = jax.random.PRNGKey(0)
    true_x_prior = mixture_prior(
        weights=data["weights"],
        means=data["means"],
        vars=data["vars"],
    )

    # Fit model
    if args.normal:
        tcup_model = model_builder(true_x_prior, ncup=True)
        PARAMS_OF_INTEREST.remove("nu")
    elif args.fixed:
        tcup_model = model_builder(
            true_x_prior,
            fixed_nu=args.fixed,
            normal_obs=args.t_obs,
        )
        PARAMS_OF_INTEREST.remove("nu")
    else:
        tcup_model = model_builder(true_x_prior, normal_obs=args.t_obs)
        if "nu" not in params:
            PARAMS_OF_INTEREST.remove("nu")

    kernel = numpyro.infer.NUTS(tcup_model)
    mcmc = numpyro.infer.MCMC(
        kernel, num_chains=1, num_warmup=1000, num_samples=L
    )

    rng_key, rng_key_ = jax.random.split(rng_key)
    results = None
    while True:
        mcmc.run(
            rng_key_,
            x_scaled=data["x_scaled"],
            y_scaled=data["y_scaled"],
            cov_x_scaled=data["cov_x_scaled"],
            dy_scaled=data["dy_scaled"],
        )
        new_results = az.from_numpyro(mcmc)

        if results is None:
            results = new_results
        else:
            results = extend_inference_data(results, new_results)

        min_ess = (
            az.ess(results, var_names=PARAMS_OF_INTEREST)
            .to_array()
            .min()
            .item()
        )

        if min_ess > L:
            break

        curr_samples = results.posterior.sizes["draw"]
        required_samples = (
            np.ceil(curr_samples * L / min_ess).astype(int) - curr_samples
        )
        print(min_ess, curr_samples, required_samples)
        mcmc.num_samples = np.clip(required_samples, 1000, 10000)
        mcmc.post_warmup_state = mcmc.last_state
        rng_key = mcmc.post_warmup_state.rng_key

        results.to_netcdf(f"{args.outfile}.checkpoint")

        if curr_samples > 1e6:
            raise RuntimeError("Max sample size exceeded")

    # Postprocess to only required MCMC samples
    samples = az.extract(
        results,
        var_names=PARAMS_OF_INTEREST,
        num_samples=L,
        rng=0,
    )

    rng_key, rng_key_ = jax.random.split(rng_key)
    post_pred = numpyro.infer.Predictive(
        tcup_model,
        build_post_pred_samples(results),
    )(
        rng_key_,
        x_scaled=data["x_scaled"],
        cov_x_scaled=data["cov_x_scaled"],
        dy_scaled=data["dy_scaled"],
    )

    for var_name, var_samples in samples.items():
        ranks = var_samples < params[var_name]
        if var_name == "beta_scaled":
            samples = samples.assign(
                {
                    f"rank_{var_name}": (
                        ("beta_scaled_dim_0",),
                        ranks.values.mean(axis=-1),
                    )
                }
            )
        else:
            samples[f"rank_{var_name}"] = ranks.values.mean(axis=-1)

    for var_name in PARAMS_OF_INTEREST:
        if var_name == "beta_scaled":
            samples = samples.assign(
                {
                    f"true_{var_name}": (
                        ("beta_scaled_dim_0",),
                        params[var_name],
                    )
                }
            )
        else:
            samples[f"true_{var_name}"] = params[var_name]

    # Add posterior predictive samples
    samples = samples.assign(
        {
            "post_pred_x_scaled": (
                ("datapoint", "beta_scaled_dim_0", "sample"),
                np.moveaxis(post_pred["x_scaled"], 0, -1),
            ),
            "post_pred_y_scaled": (
                ("datapoint", "sample"),
                np.moveaxis(post_pred["y_scaled"], 0, -1),
            ),
        }
    )

    samples = samples.reset_index(["sample", "chain", "draw"], drop=True)
    samples.to_netcdf(args.outfile)
