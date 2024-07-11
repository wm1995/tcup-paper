import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    models = ["tcup"]  # ["ncup", "tcup"]
    datasets = ["t"]  # ["outlier", "normal"]

    for dataset, model in it.product(datasets, models):
        df = pd.read_csv(f"results/{dataset}_{model}_many.csv")

        if model == "tcup":
            if dataset == "outlier":
                plot_kwargs = {"color": "maroon"}
            else:
                plot_kwargs = {"color": "lightcoral", "linestyle": "dashed"}
        else:
            if dataset == "outlier":
                plot_kwargs = {"color": "midnightblue"}
            else:
                plot_kwargs = {
                    "color": "cornflowerblue",
                    "linestyle": "dashed",
                }

        mask = df["r_hat"] < 1.1

        print(f"{(~mask).sum()} runs with r_hat > 1.1")

        # ax[0].hist(df["r_hat"], bins=25, histtype="step")
        # ax[0].set_xlabel(r"$\hat{r}$")

        plot_cdf(
            df["alpha"][mask],
            ax[0, 0],
            **plot_kwargs,
        )
        # ax[0, 0].axvline(df["alpha"][mask].mean(), ls=":", **plot_kwargs)
        ax[0, 0].set_xlabel(r"$\hat{\alpha}_{MAP}$")

        plot_cdf(
            df["beta"][mask],
            ax[0, 1],
            **plot_kwargs,
        )
        # ax[0, 1].axvline(df["beta"][mask].mean(), ls=":", **plot_kwargs)
        ax[0, 1].set_xlabel(r"$\hat{\beta}_{MAP}$")

        if model == "ncup":
            plot_cdf(
                df["sigma"][mask],
                ax[1, 0],
                **plot_kwargs,
            )
        else:
            plot_cdf(
                df["sigma_68"][mask],
                ax[1, 0],
                **plot_kwargs,
            )
        # ax[1, 0].axvline(0.2 * 1.148, c="k")  # 1.148 scales up to 68% CI
        # ax[1, 0].axvline(df["sigma_68"][mask].mean(), ls=":", **plot_kwargs)
        ax[1, 0].set_xlabel(r"$\hat{\sigma}_{{68}, MAP}$")

        if model == "tcup":
            plot_cdf(
                df["nu"][mask],
                ax[1, 1],
                **plot_kwargs,
            )
            ax[1, 1].set_xlabel(r"$\hat{\nu}_{MAP}$")

    ax[0, 0].axvline(3, c="k")
    ax[0, 1].axvline(2, c="k")
    ax[1, 0].axvline(0.2, c="k")
    ax[0, 0].set_ylim(0, 1)
    ax[0, 1].set_ylim(0, 1)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"plots/{dataset}_many_cdf.pdf")
