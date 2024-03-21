#!/usr/bin/env python
import argparse
import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from tcup.numpyro import model_builder
from tcup_paper.data.io import load_dataset

MAX_SAMPLES = 200000
SAFETY_MARGIN = 1.1
PARAMS_OF_INTEREST = ["alpha_scaled", "beta_scaled", "sigma_scaled", "nu"]

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
        raise NotImplementedError("Not implemented for numpyro yet")
    elif args.fixed:
        raise NotImplementedError("Not implemented for numpyro yet")

    rng_key = jax.random.PRNGKey(0)
    true_x_prior = mixture_prior(
        weights=data["theta_mix"],
        means=data["mu_mix"],
        vars=data["sigma_mix"],
    )
    tcup_model = model_builder(true_x_prior)
    kernel = numpyro.infer.NUTS(tcup_model)
    mcmc = numpyro.infer.MCMC(kernel, num_chains=1, num_warmup=1000, num_samples=1023)

    samples = None
    while True:
        mcmc.run(rng_key, x_scaled=data["x_scaled"], y_scaled=data["y_scaled"], cov_x_scaled=data["cov_x_scaled"], dy_scaled=data["dy_scaled"],)
        new_samples = mcmc.get_samples(group_by_chain=True)

        if samples is None:
            samples = new_samples
        else:
            for key, value in new_samples.items():
                samples[key] = np.concatenate([samples[key], value], axis=1)

        min_ess = (
            az.ess(samples, var_names=PARAMS_OF_INTEREST).to_array().min().item()
        )

        if min_ess > 1023:
            break

        curr_samples = samples["alpha_scaled"].shape[1]
        required_samples = np.ceil(curr_samples * 1023 / min_ess).astype(int) - curr_samples
        print(min_ess, curr_samples, required_samples)
        mcmc.num_samples = np.clip(required_samples, 1000, 10000)
        mcmc.post_warmup_state = mcmc.last_state
        rng_key = mcmc.post_warmup_state.rng_key

        mcmc_output = az.convert_to_inference_data(samples)
        mcmc_output.to_netcdf(f"{args.outfile}.checkpoint")

    mcmc_output.to_netcdf(args.outfile)
