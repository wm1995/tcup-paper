#!/usr/bin/env python
import json

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tcup.stan import tcup

# Set up run parameters
SEED = 24601


def load_dataset(name):
    with open(f"data/data_{name}.json", "r") as f:
        dataset = json.load(f)

    data = {key: np.array(val) for key, val in dataset["data"].items()}
    params = dataset["params"]
    params["outliers"] = np.array(params["outliers"])
    return data, params


if __name__ == "__main__":
    datasets = [
        "linear_1D0",
        "linear_1D1",
        "linear_2D0",
        "linear_2D1",
        "linear_3D0",
        "linear_3D1",
    ]

    priors = [
        "invgamma",
        "invgamma2",
        "cauchy",
        "cauchy_scaled",
        "cauchy_truncated",
        "F18",
        "F18reparam",
        "nu2",
        "nu2_principled",
        "nu2_heuristic",
        "nu2_scaled",
        "invnu",
        None,  # will be fixed to nu = 2
    ]
    models = [("tcup", prior) for prior in priors]
    models.append(("ncup", None))

    results = pd.DataFrame(
        columns=[
            "dataset",
            "model",
            "prior",
            "r_hat",
            "divergences",
            "sigma_int",
            "alpha",
            "beta[0]",
            "beta[1]",
            "beta[2]",
            "nu",
            "peak_height",
        ]
    )

    for dataset in datasets:
        data, params = load_dataset(dataset)
        for model, prior in models:
            # Fit model
            if model == "tcup" and prior is None:
                fixed_nu = data.copy()
                fixed_nu["nu"] = 2
                mcmc = tcup(fixed_nu, SEED, model, prior)
            else:
                mcmc = tcup(data, SEED, model, prior)

            # Save chains
            mcmc.to_netcdf(f"results/{dataset}_{model}_{prior}.nc")

            # Calculate summary
            summary = az.summary(mcmc)

            # Add results to table
            results.loc[len(results)] = [
                dataset,
                model,
                prior,
                summary["r_hat"].max(),
                np.sum(mcmc.sample_stats["diverging"].values),
                summary["mean"].get("sigma", None),
                summary["mean"].get("alpha", None),
                summary["mean"].get("beta[0]", None),
                summary["mean"].get("beta[1]", None),
                summary["mean"].get("beta[2]", None),
                summary["mean"].get("nu", None),
                summary["mean"].get("peak_height", None),
            ]

            # Create trace plot
            az.plot_trace(mcmc)
            plt.savefig(f"plots/trace_{dataset}_{model}_{prior}.pdf")
            plt.close()

            # Save results as we go
            results.to_csv("results/results.csv", index=False)
