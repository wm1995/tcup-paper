import argparse

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from tcup_paper.data.io import load_dataset
from tcup_paper.plot.corner import plot_corner
from tcup_paper.plot.style import apply_matplotlib_style

SEED = 42

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sigma", type=float, required=False)
    parser.add_argument("--nu", action="store_true", required=False)
    parser.add_argument("--outlier_frac", action="store_true", required=False)
    parser.add_argument("dataset", type=str)
    parser.add_argument("mcmc_1", type=str)
    parser.add_argument("mcmc_2", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    # Set style
    hdi_probs = [0.393, 0.865]
    apply_matplotlib_style()

    # Load true parameter values
    _, params = load_dataset(args.dataset)
    true_vals = {
        "alpha_rescaled": params["alpha"],
        "beta_rescaled": params["beta"],
        "sigma_rescaled": params["sigma_int"],
        "sigma_68": params["sigma_int"],
        "nu": params.get("nu"),
        "outlier_frac": params.get("outlier_frac"),
    }

    # Load mcmc data
    mcmc_1 = az.from_netcdf(args.mcmc_1)
    mcmc_2 = az.from_netcdf(args.mcmc_2)

    # Choose variables
    var_names_1 = ["alpha_rescaled", "beta_rescaled"]
    if "sigma_68" in mcmc_1["posterior"]:
        var_names_1.append("sigma_68")
        if args.nu:
            var_names_1.append("nu")
        elif args.outlier_frac:
            var_names_1.append("outlier_frac")
    else:
        var_names_1.append("sigma_rescaled")
    var_names_2 = ["alpha_rescaled", "beta_rescaled"]
    if "sigma_68" in mcmc_2["posterior"]:
        var_names_2.append("sigma_68")
        if args.nu:
            var_names_2.append("nu")
        elif args.outlier_frac:
            var_names_2.append("outlier_frac")
    else:
        var_names_2.append("sigma_rescaled")
    var_labels = {
        "alpha_rescaled": r"Intercept $\alpha$",
        "beta_rescaled": r"Gradient $\beta$",
        "sigma_rescaled": r"Int. scatter $\sigma_{68}$",
        "sigma_68": r"Int. scatter $\sigma_{68}$",
        "nu": r"Shape param. $\nu$",
        "outlier_frac": r"Outlier frac. $\omega$",
    }

    # Set styles for different datasets
    ncup_kwargs = {"color": "midnightblue"}
    fixed3_kwargs = {"color": "darkorange"}
    tcup_kwargs = {"color": "maroon"}
    if "tcup" in args.mcmc_1:
        mcmc_1_kwargs = tcup_kwargs
    elif "ncup" in args.mcmc_1:
        mcmc_1_kwargs = ncup_kwargs
    elif "fixed3" in args.mcmc_1:
        mcmc_1_kwargs = fixed3_kwargs
    if "tcup" in args.mcmc_2:
        mcmc_2_kwargs = tcup_kwargs
    elif "ncup" in args.mcmc_2:
        mcmc_2_kwargs = ncup_kwargs
    elif "fixed3" in args.mcmc_2:
        mcmc_2_kwargs = fixed3_kwargs

    # Set up histogram bins for marginal distribution
    if args.max_sigma is not None:
        sigma_bins = np.linspace(0, args.max_sigma, 50)
        bins = {
            "sigma_rescaled": sigma_bins,
            "sigma_68": sigma_bins,
        }
    else:
        bins = {}

    # Plot dataset 1
    fig, ax, bins = plot_corner(
        mcmc_1,
        var_names=var_names_1,
        var_labels=var_labels,
        true_vals=true_vals,
        marginal_kwargs=mcmc_1_kwargs,
        kde_kwargs={
            "hdi_probs": hdi_probs,
            "contour_kwargs": {
                key + "s": value for key, value in mcmc_1_kwargs.items()
            },
        },
        bins=bins,
        subplot_kwargs={"figsize": (6.32, 6.32)},
    )

    # Plot dataset 2
    plot_corner(
        mcmc_2,
        var_names=var_names_2,
        var_labels=var_labels,
        true_vals=true_vals,
        marginal_kwargs=mcmc_2_kwargs,
        kde_kwargs={
            "hdi_probs": hdi_probs,
            "contour_kwargs": {
                key + "s": value for key, value in mcmc_2_kwargs.items()
            },
        },
        bins=bins,
        fig=fig,
        ax=ax,
    )

    plt.savefig(args.output, backend="pgf")
