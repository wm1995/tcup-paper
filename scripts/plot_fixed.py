import itertools as it
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tcup_paper.plot import style


def plot_cdf(samples, ax, **plot_kwargs):
    samples = np.array(samples)
    samples.sort()
    cdf = np.cumsum(np.ones(samples.shape))
    cdf /= cdf[-1]
    return ax.plot(
        samples,
        cdf,
        **plot_kwargs,
    )


if __name__ == "__main__":
    style.apply_matplotlib_style()
    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(7.06, 2))

    models = ["ncup", "tcup"]
    datasets = ["outlier", "normal"]
    results_dir = Path("results/fixed/")
    include_prior = False

    plot_prior = it.chain({False, include_prior})
    for dataset, model, prior in it.product(datasets, models, plot_prior):
        results_path = results_dir / model / dataset / "results.nc"
        df = xr.open_dataset(
            results_path,
            group="prior" if prior else "posterior",
            engine="h5netcdf",
        )

        if model == "tcup":
            if (not include_prior and dataset == "normal") or prior:
                plot_kwargs = {"color": "lightcoral", "linestyle": "dotted"}
            else:
                plot_kwargs = {"color": "maroon"}
        elif model == "fixed":
            plot_kwargs = {"color": "orange", "linestyle": "dashed"}
        else:
            if (not include_prior and dataset == "normal") or prior:
                plot_kwargs = {
                    "color": "cornflowerblue",
                    "linestyle": "dashed",
                }
            else:
                plot_kwargs = {"color": "midnightblue"}

        plot_cdf(
            df["alpha"],
            ax[0],
            **plot_kwargs,
        )
        ax[0].set_xlabel(r"$\hat{\alpha}$")

        plot_cdf(
            df["beta"][0],
            ax[1],
            **plot_kwargs,
        )
        ax[1].set_xlabel(r"$\hat{\beta}$")

        if dataset == "gaussian_mix":
            plot_cdf(
                df["beta"][1],
                ax[2],
                **plot_kwargs,
            )
            ax[1].set_xlabel(r"$\hat{\beta}_0$")
            ax[2].set_xlabel(r"$\hat{\beta}_1$")

            plot_cdf(
                df["sigma_68"],
                ax[3],
                **plot_kwargs,
            )
            ax[3].set_xlabel(r"$\hat{\sigma}_{{68}}$")
        else:
            plot_cdf(
                df["sigma_68"],
                ax[2],
                **plot_kwargs,
            )
            ax[2].set_xlabel(r"$\hat{\sigma}_{{68}}$")

            if model == "tcup":
                plot_cdf(
                    df["nu"],
                    ax[3],
                    **plot_kwargs,
                )
                ax[3].set_xlabel(r"$\hat{\nu}$")

    if dataset == "t":
        # Set ground truth lines
        ax[0].axvline(3, c="k")
        ax[1].axvline(2, c="k")
        ax[2].axvline(0.1, c="k")
        ax[3].axvline(3, c="k")

        # Set axis limits
        ax[0].set_xlim(2.5, 3.5)
        ax[1].set_xlim(1.8, 2.2)
        ax[2].set_xlim(0, 0.5)
        ax[3].set_xlim(0, 10)
    elif dataset in ["normal", "outlier"]:
        # Set ground truth lines
        ax[0].axvline(3, c="k")
        ax[1].axvline(2, c="k")
        ax[2].axvline(0.2, c="k")

        # Set axis limits
        ax[0].set_xlim(1, 5)
        ax[1].set_xlim(1, 2.5)
        ax[2].set_xlim(0, 3)
        ax[3].set_xlim(0, 20)
    elif dataset == "laplace":
        # Set ground truth lines
        ax[0].axvline(-1, c="k")
        ax[1].axvline(0.8, c="k")
        # Laplace sigma is 0.2
        # Hence sigma_68 is 0.2 * 1.14787
        ax[2].axvline(0.2 * 1.14787, c="k")

        # Set axis limits
        ax[0].set_xlim(-2, 0)
        ax[1].set_xlim(0.5, 1.1)
        ax[2].set_xlim(0, 0.6)
        ax[3].set_xlim(0, 20)
    elif dataset == "lognormal":
        # Set ground truth lines
        ax[0].axvline(4, c="k")
        ax[1].axvline(8, c="k")
        ax[2].axvline(0.2, c="k")

        # Set axis limits
        ax[0].set_xlim(0, 10)
        ax[1].set_xlim(0, 20)
        ax[2].set_xlim(0, 40)
        ax[3].set_xlim(0, 20)
    elif dataset == "gaussian_mix":
        # Set ground truth lines
        ax[0].axvline(2, c="k")
        ax[1].axvline(3, c="k")
        ax[2].axvline(-1, c="k")
        # Scaling for sigma_68
        # sigma_int = 0.4
        # sigma_68 = 0.45859
        ax[3].axvline(0.45859, c="k")

    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[2].set_ylim(0, 1)
    ax[3].set_ylim(0, 1)
    ax[0].set_ylabel("CDF")
    fig.tight_layout()
    fig.savefig(f"plots/fixed/{dataset}_cdf.pdf")
