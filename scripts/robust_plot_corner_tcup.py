# Make rules are going to be something like
# lsa : sections/*/*.tex, sections/*/graphics/scripts/*.py, preamble.tex, references.bib
#   xelatex? lualatex? + biber + lualatex * 2
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
    t_mcmc = az.from_netcdf("tcup.nc")
    t_outlier_mcmc = az.from_netcdf("tcup_outlier.nc")
    n_mcmc = az.from_netcdf("ncup.nc")
    n_outlier_mcmc = az.from_netcdf("ncup_outlier.nc")


    fig, ax = plt.subplots(4, 4, figsize=(6.32, 6.32))
    var_names = ["alpha", "beta", "sigma", "peak_height"]
    var_labels = {"alpha": "Intercept", "beta": "Gradient", "sigma": "Intrinsic scatter", "peak_height": "Normality"}

    no_outlier_kwargs = {"color": "lightcoral", "linestyle": "dashed"}
    outlier_kwargs = {"color": "maroon"}

    for idx, var_x in enumerate(var_names):
        for idy, var_y in enumerate(var_names):
            if idx > idy:
                # Not in the corner plot, so blank it
                ax[idy, idx].axis("off")
            elif idx == idy:
                ax[idx, idx].set_yticks([])
                _, bins, _ = ax[idy, idx].hist(t_outlier_mcmc["posterior"][var_x].values.flatten(), bins=50, histtype="step", density=True, **outlier_kwargs)
                ax[idy, idx].hist(t_mcmc["posterior"][var_x].values.flatten(), bins=bins, histtype="step", density=True, **no_outlier_kwargs)
            else:
                if idx > 0:
                    plt.setp(ax[idy, idx].get_yticklabels(), visible=False)
                    ax[idy, idx].sharey(ax[idy, 0])
                else:
                    ax[idy, 0].set_ylabel(var_labels[var_y], verticalalignment="baseline")
                ax[idy, idx].sharex(ax[idx, idx])
                az.plot_kde(
                    t_mcmc["posterior"].squeeze()[var_x],
                    t_mcmc["posterior"].squeeze()[var_y],
                    ax=ax[idy, idx],
                    hdi_probs=[0.3, 0.6, 0.9],  # Plot 30%, 60% and 90% HDI contours
                    contour_kwargs={key + 's': value for key, value in no_outlier_kwargs.items()},
                    contourf_kwargs={"alpha": 0},
                )
                az.plot_kde(
                    t_outlier_mcmc["posterior"].squeeze()[var_x],
                    t_outlier_mcmc["posterior"].squeeze()[var_y],
                    ax=ax[idy, idx],
                    hdi_probs=[0.3, 0.6, 0.9],  # Plot 30%, 60% and 90% HDI contours
                    contour_kwargs={key + 's': value for key, value in outlier_kwargs.items()},
                    contourf_kwargs={"alpha": 0},
                )
            ax[-1, idx].set_xlabel(var_labels[var_x])
    fig.align_labels()
    fig.subplots_adjust(wspace=0, hspace=0)

    ax[1, 0].set_ylim((1.6, 2.3))
    ax[2, 0].set_ylim((0, 1.2))
    ax[3, 0].set_ylim((0, 1))
    ax[3, 0].set_xlim((1.8, 4.6))
    ax[3, 1].set_xlim((1.6, 2.2))
    ax[3, 2].set_xlim((0, 1.5))

    plt.savefig("corner-tcup.pdf", backend="pgf")