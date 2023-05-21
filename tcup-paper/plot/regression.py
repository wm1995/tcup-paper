import numpy as np


def plot_regression(
    rng,
    ax,
    mcmc,
    x,
    y,
    dx,
    dy,
    xlim=None,
    params=None,
    regression_kwargs=None,
    data_kwargs=None,
):
    if xlim is None:
        x_min = (x - dx).min()
        x_max = (x + dx).max()
        x_range = x_min - x_max
        xlim = (x_min - 0.05 * x_range, x_max + 0.05 * x_range)

    x_axis = np.linspace(*xlim, 200)

    inds = rng.choice(
        mcmc["posterior"].dims["chain"] * mcmc["posterior"].dims["draw"],
        size=100,
    )
    ax.plot(
        x_axis,
        [
            mcmc["posterior"]["alpha_rescaled"].values.flatten()[inds]
            + mcmc["posterior"]["beta_rescaled"].values.flatten()[inds] * x_val
            for x_val in x_axis
        ],
        **regression_kwargs,
    )
    ax.plot(
        x_axis,
        params["alpha"] + params["beta"] * x_axis,
        **data_kwargs,
    )
    ax.errorbar(
        x,
        y,
        dy,
        dx,
        "k+",
    )
    ax.set_xlim(xlim)
