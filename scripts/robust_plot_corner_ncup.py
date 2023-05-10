import json

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SEED = 42

if __name__ == "__main__":
    # Set matplotlib style
    preamble = r"""
    \usepackage{unicode-math}
    \setmainfont[
        BoldFont = texgyrepagella-bold.otf,
        ItalicFont = texgyrepagella-italic.otf,
        BoldItalicFont = texgyrepagella-bolditalic.otf
    ]{texgyrepagella-regular.otf}
    \setmathfont{texgyrepagella-math.otf}
    \setmathfont{Asana-Math.otf}[range={\mitxi}]      % Use Asana for xi
    \setmathfont{latinmodern-math.otf}[range=\symcal] % Use Latin Modern for mathcal
    """
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["pgf.preamble"] = preamble
    mpl.rcParams["pgf.rcfonts"] = False
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"

    # Load mcmc data
    t_mcmc = az.from_netcdf("results/normal_tcup.nc")
    t_outlier_mcmc = az.from_netcdf("results/outlier_tcup.nc")
    n_mcmc = az.from_netcdf("results/normal_ncup.nc")
    n_outlier_mcmc = az.from_netcdf("results/outlier_ncup.nc")

    var_names_ncup = ["alpha_rescaled", "beta_rescaled", "sigma_rescaled"]
    var_names_tcup = ["alpha_rescaled", "beta_rescaled", "sigma_68"]
    var_labels = {
        "alpha_rescaled": "Intercept",
        "beta_rescaled": "Gradient",
        "sigma_rescaled": "Intrinsic scatter",
    }
    hdi_probs = [0.393, 0.865]
    fig, ax = plt.subplots(
        len(var_names_ncup), len(var_names_ncup), figsize=(6.32, 6.32)
    )

    no_outlier_kwargs = {"color": "cornflowerblue", "linestyle": "dashed"}
    outlier_kwargs = {"color": "midnightblue"}
    tcup_kwargs = {"color": "maroon"}
    tcup_no_outlier_kwargs = {"color": "lightcoral", "linestyle": "dashed"}

    for idx, (var_x_ncup, var_x_tcup) in enumerate(
        zip(var_names_ncup, var_names_tcup)
    ):
        for idy, (var_y_ncup, var_y_tcup) in enumerate(
            zip(var_names_ncup, var_names_tcup)
        ):
            if idx > idy:
                # Not in the corner plot, so blank it
                ax[idy, idx].axis("off")
            elif idx == idy:
                ax[idx, idx].set_yticks([])
                if var_x_ncup == "sigma_rescaled":
                    _, bins, _ = ax[idy, idx].hist(
                        n_outlier_mcmc["posterior"][
                            var_x_ncup
                        ].values.flatten(),
                        bins=50,
                        range=(0, 7),
                        histtype="step",
                        **outlier_kwargs
                    )
                else:
                    _, bins, _ = ax[idy, idx].hist(
                        n_outlier_mcmc["posterior"][
                            var_x_ncup
                        ].values.flatten(),
                        bins=50,
                        histtype="step",
                        **outlier_kwargs
                    )
                ax[idy, idx].hist(
                    n_mcmc["posterior"][var_x_ncup].values.flatten(),
                    bins=bins,
                    histtype="step",
                    **no_outlier_kwargs
                )
                ax[idy, idx].hist(
                    t_outlier_mcmc["posterior"][var_x_tcup].values.flatten(),
                    bins=bins,
                    histtype="step",
                    **tcup_kwargs
                )
                # ax[idy, idx].hist(t_mcmc["posterior"][var_x].values.flatten(), bins=bins, histtype="step", **tcup_no_outlier_kwargs)
            else:
                if idx > 0:
                    plt.setp(ax[idy, idx].get_yticklabels(), visible=False)
                    ax[idy, idx].sharey(ax[idy, 0])
                else:
                    ax[idy, 0].set_ylabel(
                        var_labels[var_y_ncup], verticalalignment="baseline"
                    )
                ax[idy, idx].sharex(ax[idx, idx])
                az.plot_kde(
                    n_mcmc["posterior"].squeeze()[var_x_ncup],
                    n_mcmc["posterior"].squeeze()[var_y_ncup],
                    ax=ax[idy, idx],
                    hdi_probs=hdi_probs,
                    contour_kwargs={
                        key + "s": value
                        for key, value in no_outlier_kwargs.items()
                    },
                    contourf_kwargs={"alpha": 0},
                )
                az.plot_kde(
                    n_outlier_mcmc["posterior"].squeeze()[var_x_ncup],
                    n_outlier_mcmc["posterior"].squeeze()[var_y_ncup],
                    ax=ax[idy, idx],
                    hdi_probs=hdi_probs,
                    contour_kwargs={
                        key + "s": value
                        for key, value in outlier_kwargs.items()
                    },
                    contourf_kwargs={"alpha": 0},
                )
                az.plot_kde(
                    t_outlier_mcmc["posterior"].squeeze()[var_x_tcup],
                    t_outlier_mcmc["posterior"].squeeze()[var_y_tcup],
                    ax=ax[idy, idx],
                    hdi_probs=hdi_probs,
                    contour_kwargs={
                        key + "s": value for key, value in tcup_kwargs.items()
                    },
                    contourf_kwargs={"alpha": 0},
                )
                # az.plot_kde(
                #     t_mcmc["posterior"].squeeze()[var_x],
                #     t_mcmc["posterior"].squeeze()[var_y],
                #     ax=ax[idy, idx],
                #     hdi_probs=[0.3, 0.6, 0.9],  # Plot 30%, 60% and 90% HDI contours
                #     contour_kwargs={key + 's': value for key, value in tcup_no_outlier_kwargs.items()},
                #     contourf_kwargs={"alpha": 0},
                # )
            ax[-1, idx].set_xlabel(var_labels[var_x_ncup])
    fig.align_labels()
    fig.subplots_adjust(wspace=0, hspace=0)

    ax[1, 0].set_ylim((1.1, 2.5))
    ax[2, 0].set_ylim((0, 5))
    ax[2, 0].set_xlim((-2, 7))
    ax[2, 1].set_xlim((0.8, 2.7))
    ax[2, 2].set_xlim((0, 7))

    plt.savefig("corner-ncup.pdf", backend="pgf")
