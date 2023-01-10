from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

if __name__ == "__main__":
    results = pd.DataFrame(
        columns=[
            "dataset",
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

    for filepath in Path("results/").glob("*.nc"):
        # Trim extension from filename
        filename = filepath.name[:-3]
        dataset = "".join(filename.split("_")[:2])
        prior = "".join(filename.split("_")[2:])
        mcmc = az.from_netcdf(filepath)
        summary = az.summary(mcmc)

        # Add results to table
        results.loc[len(results)] = [
            dataset,
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

    results.sort_values(["dataset", "prior"], inplace=True)
    results.to_csv("results/results.csv", index=False)
