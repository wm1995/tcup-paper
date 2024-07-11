import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


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
    fig, ax = plt.subplots(2, 2)

    models = ["ncup", "tcup"]
    datasets = ["outlier", "normal"]
    models = ["tcup"]  # ["ncup", "tcup"]
    datasets = [
        "../../tcup-paper-fi-no-scaling/tcup-paper-final-iteration/results/t",
        "t",
    ]  # ["outlier", "normal"]

    for dataset, model in it.product(datasets, models):
        df = xr.open_dataset(f"results/{dataset}_{model}_many_samples.nc")

        if model == "tcup":
            if dataset in ["prior", "normal"]:
                plot_kwargs = {"color": "lightcoral", "linestyle": "dashed"}
            elif ".." in dataset:
                plot_kwargs = {"color": "green"}  # maroon"}
            else:
                plot_kwargs = {"color": "maroon"}
        else:
            if dataset in ["prior", "normal"]:
                plot_kwargs = {
                    "color": "cornflowerblue",
                    "linestyle": "dashed",
                }
            else:
                plot_kwargs = {"color": "midnightblue"}

        plot_cdf(
            df["alpha"],
            ax[0, 0],
            **plot_kwargs,
        )
        # ax[0, 0].axvline(df["alpha"][mask].mean(), ls=":", **plot_kwargs)
        ax[0, 0].set_xlabel(r"$\hat{\alpha}$")

        plot_cdf(
            df["beta"][0],
            ax[0, 1],
            **plot_kwargs,
        )
        # ax[0, 1].axvline(df["beta"][mask].mean(), ls=":", **plot_kwargs)
        ax[0, 1].set_xlabel(r"$\hat{\beta}$")

        plot_cdf(
            df["sigma_68"],
            ax[1, 0],
            **plot_kwargs,
        )
        # ax[1, 0].axvline(0.2 * 1.148, c="k")  # 1.148 scales up to 68% CI
        # ax[1, 0].axvline(df["sigma_68"][mask].mean(), ls=":", **plot_kwargs)
        ax[1, 0].set_xlabel(r"$\hat{\sigma}_{{68}}$")

        if model == "tcup":
            plot_cdf(
                df["nu"],
                ax[1, 1],
                **plot_kwargs,
            )
            ax[1, 1].set_xlabel(r"$\hat{\nu}$")

    ax[0, 0].axvline(3, c="k")
    ax[0, 1].axvline(2, c="k")
    ax[1, 0].axvline(0.1, c="k")
    ax[1, 1].axvline(3, c="k")
    ax[0, 0].set_xlim(2, 4)
    ax[0, 1].set_xlim(1.5, 2.5)
    ax[1, 0].set_xlim(0, 1)
    ax[1, 1].set_xlim(0, 20)
    # ax[0, 0].set_xlim(-2, 6)
    # ax[0, 1].set_xlim(0, 3)
    # ax[1, 0].set_xlim(0, 4)
    # ax[1, 1].set_xlim(0, 50)
    ax[0, 0].set_ylim(0, 1)
    ax[0, 1].set_ylim(0, 1)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"plots/{dataset}_samples_cdf.pdf")
