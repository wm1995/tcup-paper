import argparse
import warnings

import arviz as az
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats as sps
from tqdm import trange


def kde_map(samples, x0=None):
    samples = samples.to_array().values
    n_vars = samples.shape[0]
    samples = samples.reshape(n_vars, -1)
    if x0 is None:
        x0 = np.zeros(n_vars)

    kernel = sps.gaussian_kde(samples)
    soln = scipy.optimize.minimize(lambda x: -kernel.logpdf(x), x0)

    assert soln.success
    return soln.x


# This is an implementation of the half-sample mode algorithm
# Translated from an R function in Statomics
# R source: http://dawningrealm.org/stats/statomics/index.html
# I'm guessing there's a way of vectorising this but this works for now
def hsm(x):
    y = np.sort(x.flatten())
    while y.size > 4:
        m = np.ceil(y.size / 2).astype(int)
        w_min = y[-1] - y[0]
        for i in range(y.size - m):
            w = y[i + m] - y[i]
            if w <= w_min:
                w_min = w
                j = i
        if w == 0:
            y = y[j]
        else:
            y = y[j : j + m - 1]
    if y.size == 3:
        z = 2 * y[1] - y[0] - y[2]
        assert np.isfinite(z)
        if z < 0:
            return np.mean(y[:1])
        elif z > 0:
            return np.mean(y[1:])
        else:
            return y[1]
    else:
        return np.mean(y)


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

    results = pd.DataFrame(columns=["r_hat", "diverging"] + var_names)

    for idx in trange(args.num_repeats):
        try:
            mcmc = az.from_netcdf(
                f"results/fixed/{args.model}/{args.dataset}/{idx+1}.nc"
            )
        except FileNotFoundError:
            warnings.warn("File not found")
            continue

        diagnostics = az.summary(mcmc, kind="diagnostics")
        summary = az.summary(mcmc, var_names=var_names, stat_focus="median")

        # map_estimate = kde_map(
        #     mcmc.posterior[["alpha", "beta", "sigma_68", "nu"]],
        #     [3, 2, 0.1, 3],
        # )

        # # add results to table
        # results.loc[len(results)] = [
        #     diagnostics["r_hat"].max(),
        #     np.sum(mcmc.sample_stats["diverging"].values),
        # ] + map_estimate.tolist()

        # Add results to table
        if args.model in ["tcup", "tobs"]:
            results.loc[len(results)] = [
                diagnostics["r_hat"].max(),
                np.sum(mcmc.sample_stats["diverging"].values),
                summary["median"].get("alpha", None),
                summary["median"].get("beta[0]", None),
                summary["median"].get("sigma_68", None),
                summary["median"].get("nu", None),
            ]
        elif args.model == "ncup":
            results.loc[len(results)] = [
                diagnostics["r_hat"].max(),
                np.sum(mcmc.sample_stats["diverging"].values),
                summary["median"].get("alpha", None),
                summary["median"].get("beta[0]", None),
                summary["median"].get("sigma", None),
            ]
        else:
            results.loc[len(results)] = [
                diagnostics["r_hat"].max(),
                np.sum(mcmc.sample_stats["diverging"].values),
                summary["median"].get("alpha", None),
                summary["median"].get("beta[0]", None),
                summary["median"].get("sigma_68", None),
            ]

    results.to_csv(f"results/{args.dataset}_{args.model}_many.csv")
