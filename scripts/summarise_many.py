from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from tqdm import trange


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
    var_names = [
        "alpha_rescaled",
        "beta_rescaled",
        "sigma_68",
        "nu",
    ]
    results = pd.DataFrame(columns=["r_hat", "diverging"] + var_names)

    for idx in trange(550):
        mcmc = az.from_netcdf(f"results/repeats/laplace_tcup_{idx}.nc")
        diagnostics = az.summary(mcmc, kind="diagnostics")
        summary = az.summary(
            mcmc, var_names=var_names, stat_funcs={"map": hsm}
        )

        # Add results to table
        results.loc[len(results)] = [
            diagnostics["r_hat"].max(),
            np.sum(mcmc.sample_stats["diverging"].values),
            summary["map"].get("alpha_rescaled", None),
            summary["map"].get("beta_rescaled", None),
            summary["map"].get("sigma_68", None),
            summary["map"].get("nu", None),
        ]

    results.to_csv("results/laplace_tcup_many.csv")
