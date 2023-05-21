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
    t_outlier_mcmc = az.from_netcdf("results/outlier_tcup.nc")
    n_mcmc = az.from_netcdf("results/normal_ncup.nc")
    n_outlier_mcmc = az.from_netcdf("results/outlier_ncup.nc")

    # Choose variables
    var_names_ncup = ["alpha_rescaled", "beta_rescaled", "sigma_rescaled"]
    var_names_tcup = ["alpha_rescaled", "beta_rescaled", "sigma_68"]
    var_labels = {
        "alpha_rescaled": r"Intercept $\alpha$",
        "beta_rescaled": r"Gradient $\beta$",
        "sigma_rescaled": r"Intrinsic scatter $\sigma_{int}$",
    }

    # Set styles for different datasets
    no_outlier_kwargs = {"color": "cornflowerblue", "linestyle": "dashed"}
    outlier_kwargs = {"color": "midnightblue"}
    tcup_kwargs = {"color": "maroon"}

    # Set up histogram bins for marginal distribution
    sigma_bins = np.linspace(0, 7, 50)
    bins = {
        "sigma_rescaled": sigma_bins,
        "sigma_68": sigma_bins,
    }

    # Set up plot
    fig, ax = plt.subplots(3, 3, figsize=(6.32, 6.32))

    # Plot outlier ncup mcmc first (to set scales for other plots)
    _, _, bins = plot_corner(
        fig,
        ax,
        n_outlier_mcmc,
        var_names=var_names_ncup,
        var_labels=var_labels,
        true_vals=true_vals,
        marginal_kwargs=outlier_kwargs,
        kde_kwargs={
            "hdi_probs": hdi_probs,
            "contour_kwargs": {
                key + "s": value for key, value in outlier_kwargs.items()
            },
        },
        bins=bins,
    )

    # Plot outlier-free ncup mcmc
    plot_corner(
        fig,
        ax,
        n_mcmc,
        var_names=var_names_ncup,
        true_vals=true_vals,
        marginal_kwargs=no_outlier_kwargs,
        kde_kwargs={
            "hdi_probs": hdi_probs,
            "contour_kwargs": {
                key + "s": value for key, value in no_outlier_kwargs.items()
            },
        },
        bins=bins,
    )

    # Plot outlier tcup mcmc
    plot_corner(
        fig,
        ax,
        t_outlier_mcmc,
        var_names=var_names_tcup,
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

    ax[1, 0].set_ylim((0.9, 2.5))
    ax[2, 0].set_ylim((0, 5))
    ax[2, 0].set_xlim((-2, 8))
    ax[2, 1].set_xlim((0.8, 2.7))
    ax[2, 2].set_xlim((0, 7))

    plt.savefig("plots/corner_ncup.pdf", backend="pgf")
