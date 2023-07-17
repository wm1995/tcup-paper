import itertools as it
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    fig, ax = plt.subplots(2, 2)

    models = ["ncup", "tcup"]
    datasets = ["outlier", "normal"]

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

        ax[0, 0].hist(
            df["alpha_rescaled"][mask],
            bins=25,
            histtype="step",
            density=True,
            cumulative=True,
            **plot_kwargs,
        )
        ax[0, 0].axvline(
            df["alpha_rescaled"][mask].mean(), ls=":", **plot_kwargs
        )
        ax[0, 0].set_xlabel(r"$\hat{\alpha}_{MAP}$")

        ax[0, 1].hist(
            df["beta_rescaled"][mask],
            bins=25,
            histtype="step",
            density=True,
            cumulative=True,
            **plot_kwargs,
        )
        ax[0, 1].axvline(
            df["beta_rescaled"][mask].mean(), ls=":", **plot_kwargs
        )
        ax[0, 1].set_xlabel(r"$\hat{\beta}_{MAP}$")

        if model == "tcup":
            ax[1, 0].hist(
                df["sigma_rescaled"][mask],
                bins=25,
                histtype="step",
                density=True,
                cumulative=True,
                **plot_kwargs,
            )
            # ax[1, 0].axvline(0.2 * 1.148, c="k")  # 1.148 scales up to 68% CI
            ax[1, 0].axvline(
                df["sigma_rescaled"][mask].mean(), ls=":", **plot_kwargs
            )
            ax[1, 0].set_xlabel(r"$\hat{\sigma}_{{\rm int}, MAP}$")

            ax[1, 1].hist(
                df["nu"][mask],
                bins=25,
                histtype="step",
                density=True,
                cumulative=True,
                **plot_kwargs,
            )
            ax[1, 1].set_xlabel(r"$\hat{\nu}_{MAP}$")
        else:
            ax[1, 0].hist(
                df["sigma_rescaled"][mask],
                bins=25,
                histtype="step",
                density=True,
                cumulative=True,
                **plot_kwargs,
            )
            # ax[1, 0].axvline(0.2 * 1.148, c="k")  # 1.148 scales up to 68% CI
            ax[1, 0].axvline(
                df["sigma_rescaled"][mask].mean(), ls=":", **plot_kwargs
            )
            ax[1, 0].set_xlabel(r"$\hat{\sigma}_{{\rm int}, MAP}$")

    ax[0, 0].axvline(3, c="k")
    ax[0, 1].axvline(2, c="k")
    ax[1, 0].axvline(0.2, c="k")
    ax[0, 0].set_ylim(0, 1)
    ax[0, 1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig("plots/outlier_many_cdf.svg")
