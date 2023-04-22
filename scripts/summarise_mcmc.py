from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

if __name__ == "__main__":
    results = pd.DataFrame(
        columns=[
            "dataset",
            "model",
            "r_hat",
            "divergences",
            "sigma_int",
            "alpha",
            "beta[0]",
            "beta[1]",
            "beta[2]",
            "nu",
        ]
    )

    for filepath in Path("results/").glob("*.nc"):
        # Trim extension from filename
        filename = filepath.name[:-3]
        dataset = "".join(filename.split("_")[:-1])
        model = "".join(filename.split("_")[-1])
        mcmc = az.from_netcdf(filepath)
        summary = az.summary(mcmc)

        # Add results to table
        results.loc[len(results)] = [
            dataset,
            model,
            summary["r_hat"].max(),
            np.sum(mcmc.sample_stats["diverging"].values),
            summary["mean"].get("sigma", None),
            summary["mean"].get("alpha_rescaled", None),
            summary["mean"].get("beta_rescaled[0]", None),
            summary["mean"].get("beta_rescaled[1]", None),
            summary["mean"].get("beta_rescaled[2]", None),
            summary["mean"].get("nu", None),
        ]

    results.sort_values(["dataset", "model"], inplace=True)
    results.to_csv("results/results.csv", index=False)
