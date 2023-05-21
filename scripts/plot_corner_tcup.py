import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from tcup_paper.data.io import load_dataset
from tcup_paper.plot.corner import plot_corner
from tcup_paper.plot.style import apply_matplotlib_style

SEED = 42

if __name__ == "__main__":
    # Set style
    hdi_probs = [0.393, 0.865]
    apply_matplotlib_style()

    # Load true parameter values
    _, params = load_dataset("data/normal.json")
    true_vals = {
        "alpha_rescaled": params["alpha"],
        "beta_rescaled": params["beta"],
        "sigma_rescaled": params["sigma_int"],
        "sigma_68": params["sigma_int"],
    }

    # Load mcmc data
    t_mcmc = az.from_netcdf("results/normal_tcup.nc")
    t_outlier_mcmc = az.from_netcdf("results/outlier_tcup.nc")

    # Choose variables
    var_names = ["alpha_rescaled", "beta_rescaled", "sigma_68", "nu"]
    var_labels = {
        "alpha_rescaled": r"Intercept $\alpha$",
        "beta_rescaled": r"Gradient $\beta$",
        "sigma_68": r"Int. scatter $\sigma_{int}$",
        "nu": r"Shape param. $\nu$",
    }

    # Set styles for different datasets
    tcup_kwargs = {"color": "maroon"}
    tcup_no_outlier_kwargs = {"color": "lightcoral", "linestyle": "dashed"}

    # Set up histogram bins for marginal distribution
    bins = {
        "sigma_68": np.linspace(0, 3, 50),
        "nu": np.linspace(0, 25, 50),
    }

    # Set up plot
    fig, ax = plt.subplots(4, 4, figsize=(6.32, 6.32))

    # Plot outlier tcup mcmc first (to set scale for other plot)
    _, _, bins = plot_corner(
        fig,
        ax,
        t_outlier_mcmc,
        var_names=var_names,
        var_labels=var_labels,
        true_vals=true_vals,
        marginal_kwargs=tcup_kwargs,
        kde_kwargs={
            "hdi_probs": hdi_probs,
            "contour_kwargs": {
                key + "s": value for key, value in tcup_kwargs.items()
            },
        },
        bins=bins,
    )

    # Plot outlier tcup mcmc
    plot_corner(
        fig,
        ax,
        t_mcmc,
        var_names=var_names,
        true_vals=true_vals,
        marginal_kwargs=tcup_no_outlier_kwargs,
        kde_kwargs={
            "hdi_probs": hdi_probs,
            "contour_kwargs": {
                key + "s": value
                for key, value in tcup_no_outlier_kwargs.items()
            },
        },
        bins=bins,
    )

    ax[1, 0].set_ylim((1.4, 2.7))
    ax[2, 0].set_ylim((0, 2.4))
    ax[3, 0].set_ylim((0, 25))
    ax[3, 0].set_xlim((0.2, 4.8))
    ax[3, 1].set_xlim((1.4, 2.7))
    ax[3, 2].set_xlim((0, 2.4))
    ax[3, 3].set_xlim((0, 25))

    plt.savefig("plots/corner_tcup.pdf", backend="pgf")
