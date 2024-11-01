import argparse

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from tcup_paper.data.io import load_dataset
from tcup_paper.plot.corner import plot_corner
from tcup_paper.plot.style import apply_matplotlib_style

SEED = 42

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-i", "--mcmc-file", action="append", required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument(
        "--range",
        action="append",
        nargs=3,
        metavar=("var_name", "min", "max"),
        required=False,
    )
    parser.add_argument(
        "--ticker",
        action="append",
        nargs=3,
        metavar=("var_name", "base", "offset"),
        required=False,
    )
    parser.add_argument(
        "--var-names",
        nargs="+",
        required=False,
    )
    parser.add_argument("--single-column", action="store_true")
    args = parser.parse_args()

    # Set style
    hdi_probs = [0.393, 0.865]
    apply_matplotlib_style()

    # Load true parameter values if dataset is specified
    true_vals = {}
    if args.dataset:
        _, params = load_dataset(args.dataset)
        true_vals = {
            "alpha": params["alpha"],
            "beta": params["beta"],
            "sigma_68": params.get("sigma_68", params.get("sigma_int")),
            "nu": params.get("nu"),
            "outlier_frac": params.get("outlier_frac"),
        }
        if np.array(true_vals["beta"]).ndim != 1:
            true_vals["beta"] = [true_vals["beta"]]

    fig = None
    ax = None

    first_pass = True

    # Set up histogram bins for marginal distribution
    bins = {}
    if args.range is not None:
        for var_name, lower, upper in args.range:
            var_bins = np.linspace(float(lower), float(upper), 50)
            if var_name == "sigma":
                bins["sigma_68"] = var_bins
                bins["sigma"] = var_bins
            elif var_name[:4] == "beta":
                bins[var_name] = var_bins
            elif var_name == "alpha":
                bins["alpha"] = var_bins
            else:
                bins[var_name] = var_bins

    # Loop over inputs
    for mcmc_file in args.mcmc_file:
        mcmc = az.from_netcdf(mcmc_file)

        if args.var_names is None:
            # Choose variables
            var_names = ["alpha", "beta", "sigma_68"]
        else:
            var_names = []
            for var_name in args.var_names:
                if var_name == "alpha":
                    var_names.append("alpha")
                elif var_name == "beta":
                    var_names.append("beta")
                elif var_name == "sigma_68":
                    var_names.append("sigma_68")
                elif var_name == "nu":
                    if "tcup" in mcmc_file:
                        var_names.append("nu")
                elif var_name == "outlier_frac":
                    if "tcup" in mcmc_file:
                        var_names.append("outlier_frac")
                elif var_name == "sigma_int":
                    var_names.append("sigma")
                else:
                    raise ValueError("Unrecognised variable")

        if first_pass:
            first_vars = var_names

        if args.single_column:
            var_labels = {
                "alpha": r"$\alpha$",
                "beta": r"$\beta$",
                "sigma": r"$\sigma_{\rm int}$",
                "sigma_68": r"$\sigma_{68}$",
                "nu": r"$\nu$",
                "outlier_frac": r"$\omega$",
            }
            subplot_kwargs = {"figsize": (10 / 3, 10 / 3)}
        else:
            var_labels = {
                "alpha": r"Intercept $\alpha$",
                "beta": r"Gradient $\beta$",
                "sigma": r"Int. scatter $\sigma_{\rm int}$",
                "sigma_68": r"Int. scatter $\sigma_{68}$",
                "nu": r"Shape param. $\nu$",
                "outlier_frac": r"Outlier frac. $\omega$",
            }
            subplot_kwargs = {"figsize": (7.06, 7.06)}

        # Set styles for different datasets
        if "/normal/" in mcmc_file:
            ncup_kwargs = {"color": "cornflowerblue", "linestyle": "dashed"}
            fixed3_kwargs = {"color": "orange", "linestyle": "dashed"}
            tcup_kwargs = {"color": "lightcoral", "linestyle": "dashed"}
        else:
            ncup_kwargs = {"color": "midnightblue"}
            fixed3_kwargs = {"color": "darkorange"}
            tcup_kwargs = {"color": "maroon"}
            linmix_kwargs = {"color": "cornflowerblue"}
        if "tcup" in mcmc_file:
            mcmc_kwargs = tcup_kwargs
        elif "ncup" in mcmc_file:
            mcmc_kwargs = ncup_kwargs
        elif "linmix" in mcmc_file:
            mcmc_kwargs = linmix_kwargs
        elif "fixed/fixed" in mcmc_file:
            mcmc_kwargs = fixed3_kwargs

        # Plot dataset 1
        fig, ax, bins = plot_corner(
            mcmc,
            var_names=var_names,
            var_labels=var_labels,
            true_vals=true_vals,
            marginal_kwargs=mcmc_kwargs,
            kde_kwargs={
                "hdi_probs": hdi_probs,
                "contour_kwargs": {
                    key + "s": value for key, value in mcmc_kwargs.items()
                },
            },
            true_kwargs={
                "alpha": 0.4,
                "dashes": (3, 3),
            },
            bins=bins,
            subplot_kwargs=subplot_kwargs,
            fig=fig,
            ax=ax,
        )

        if first_pass:
            true_vals = {}
            first_pass = False

    N_vars = len(ax)
    ax_vars = []
    for var_name in first_vars:
        if "beta" in var_name:
            N_beta_plots = N_vars - len(first_vars) + 1
            ax_vars += [f"beta_{idx}" for idx in range(N_beta_plots)]
        else:
            ax_vars += [var_name]

    locators = {}
    if args.ticker is not None:
        for var_name, base, offset in args.ticker:
            locators[var_name] = plticker.MultipleLocator(
                base=float(base),
                offset=float(offset),
            )

    for subplot_ax, var_name in zip(ax[-1, :], ax_vars):
        curr_bins = bins[var_name]
        subplot_ax.set_xlim(curr_bins.min(), curr_bins.max())
        if var_name in locators:
            subplot_ax.xaxis.set_major_locator(locators[var_name])

    for subplot_ax, var_name in zip(ax[1:, 0], ax_vars[1:]):
        curr_bins = bins[var_name]
        subplot_ax.set_ylim(curr_bins.min(), curr_bins.max())
        if var_name in locators:
            subplot_ax.yaxis.set_major_locator(locators[var_name])
            
    if "park" in args.mcmc_file[0]:
        ax[1, 0].set_ylabel(r"$\beta$")
        ax[2, 0].set_ylabel(r"$\gamma$")
        ax[-1, 1].set_xlabel(r"$\beta$")
        ax[-1, 2].set_xlabel(r"$\gamma$")

    plt.savefig(args.output)
