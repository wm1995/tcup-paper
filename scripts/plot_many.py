import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("results/laplace_tcup_many.csv")

    fig, ax = plt.subplots(2, 2)

    mask = df["r_hat"] < 1.1

    print(f"{(~mask).sum()} runs with r_hat > 1.1")

    # ax[0].hist(df["r_hat"], bins=25, histtype="step")
    # ax[0].set_xlabel(r"$\hat{r}$")

    ax[0, 0].hist(df["alpha_rescaled"][mask], bins=25, histtype="step")
    ax[0, 0].axvline(-1, c="k")
    ax[0, 0].axvline(df["alpha_rescaled"][mask].mean(), ls=":")
    ax[0, 0].set_xlabel(r"$\hat{\alpha}_{MAP}$")

    ax[0, 1].hist(df["beta_rescaled"][mask], bins=25, histtype="step")
    ax[0, 1].axvline(0.8, c="k")
    ax[0, 1].axvline(df["beta_rescaled"][mask].mean(), ls=":")
    ax[0, 1].set_xlabel(r"$\hat{\beta}_{MAP}$")

    ax[1, 0].hist(df["sigma_68"][mask], bins=25, histtype="step")
    ax[1, 0].axvline(0.2 * 1.148, c="k")  # 1.148 scales up to 68% CI
    ax[1, 0].axvline(df["sigma_68"][mask].mean(), ls=":")
    ax[1, 0].set_xlabel(r"$\hat{\sigma}_{68, MAP}$")

    ax[1, 1].hist(df["nu"][mask], bins=25, histtype="step")
    ax[1, 1].set_xlabel(r"$\hat{\nu}_{MAP}$")

    fig.tight_layout()
    fig.savefig("plots/laplace_many.pdf")
