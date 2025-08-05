import itertools
from typing import Optional

import arviz as az
import fastkde
import matplotlib.pyplot as plt
import numpy as np


def plot_kde(x, y, ax, **kde_kwargs):
    hdi_probs = kde_kwargs.get("hdi_probs")
    contour_kwargs = kde_kwargs.get("contour_kwargs", {})

    log_axes = [
        x.name in ["sigma_68", "nu"],
        y.name in ["sigma_68", "nu"],
    ]
    pdf = fastkde.pdf(x.values, y.values, log_axes=log_axes)

    xx, yy = np.meshgrid(*[pdf[dim] for dim in pdf.dims[::-1]])
    if hdi_probs is not None:
        levels = az.stats.density_utils._find_hdi_contours(
            pdf,
            hdi_probs,
        )
        levels.sort()
    else:
        levels = None

    ax.contour(xx, yy, pdf, levels=levels, **contour_kwargs)


def plot_corner(
    mcmc: az.InferenceData,
    var_names: list[str],
    var_labels: Optional[dict[str, str]] = None,
    true_vals: Optional[dict] = None,
    marginal_kwargs: Optional[dict] = None,
    kde_kwargs: Optional[dict] = None,
    true_kwargs: Optional[dict] = None,
    bins: int | np.ndarray = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    subplot_kwargs: Optional[dict] = None,
):
    if var_labels is None:
        var_labels = {}

    if true_vals is None:
        true_vals = {}

    if marginal_kwargs is None:
        marginal_kwargs = {}
    marginal_kwargs.setdefault("bins", 50)

    if kde_kwargs is None:
        kde_kwargs = {}
    kde_kwargs.setdefault("contourf_kwargs", {"alpha": 0})

    if true_kwargs is None:
        true_kwargs = {}
    true_kwargs.setdefault("color", "black")
    true_kwargs.setdefault("linestyle", "dashed")
    true_kwargs.setdefault("dashes", (5, 5))
    true_kwargs.setdefault("alpha", 0.2)

    if bins is None:
        bins = {}

    # Flatten the MCMC chains/draws into a single dimension
    pooled_samples = mcmc["posterior"].stack(pooled=["chain", "draw"])

    # Work out how many variables have multiple dimensions
    # and deduce corner plot size
    dims = pooled_samples.sizes
    var_indices = []
    var_names_unpacked = []
    for var_name in var_names:
        var_dim = len(pooled_samples[var_name].dims)
        if var_dim == 1:
            var_indices += [tuple()]
            var_names_unpacked += [var_name]
        else:
            dim_generators = []
            for dim_name in pooled_samples[var_name].dims:
                if dim_name != "pooled":
                    dim_generators.append(range(dims[dim_name]))
            indices = list(itertools.product(*dim_generators))
            var_indices += indices
            var_names_unpacked += [var_name] * len(indices)

    N_plots = len(var_names_unpacked)
    if fig is None and ax is None:
        fig, ax = plt.subplots(N_plots, N_plots, **subplot_kwargs)
    else:
        assert fig is not None
        assert ax is not None
        assert len(ax) >= N_plots and len(ax[0]) >= N_plots

    for ax_idx, (var_x, var_idx) in enumerate(
        zip(var_names_unpacked, var_indices)
    ):
        if var_x not in pooled_samples:
            continue

        bins_key = var_x + "".join([f"_{x_coord}" for x_coord in var_idx])

        for ax_idy, (var_y, var_idy) in enumerate(
            zip(var_names_unpacked, var_indices)
        ):
            if var_y not in pooled_samples:
                continue
            if ax_idx > ax_idy:
                # Not in the corner plot, so blank it
                ax[ax_idy, ax_idx].axis("off")
            elif ax_idx == ax_idy:
                ax[ax_idy, ax_idx].set_yticks([])
                if bins_key in bins:
                    ax[ax_idy, ax_idx].hist(
                        pooled_samples[var_x][var_idx].values,
                        bins=bins[bins_key],
                        histtype="step",
                        density=True,
                        **{
                            key: value
                            for key, value in marginal_kwargs.items()
                            if key != "bins"
                        },
                    )
                else:
                    _, curr_bins, _ = ax[ax_idy, ax_idx].hist(
                        pooled_samples[var_x][var_idx].values,
                        histtype="step",
                        density=True,
                        **marginal_kwargs,
                    )
                    bins[bins_key] = curr_bins
                if true_vals.get(var_x) is not None:
                    if len(var_idx) > 0:
                        ax[ax_idy, ax_idx].axvline(
                            np.array(true_vals[var_x])[var_idx], **true_kwargs
                        )
                    else:
                        ax[ax_idy, ax_idx].axvline(
                            true_vals[var_x], **true_kwargs
                        )
            else:
                if ax_idx > 0:
                    plt.setp(
                        ax[ax_idy, ax_idx].get_yticklabels(), visible=False
                    )
                    ax[ax_idy, ax_idx].sharey(ax[ax_idy, 0])
                elif var_y in var_labels:
                    var_y_suffix = "".join(
                        [f"$_{y_coord}$" for y_coord in var_idy]
                    )
                    ax[ax_idy, 0].set_ylabel(
                        var_labels[var_y] + var_y_suffix,
                        verticalalignment="baseline",
                    )
                ax[ax_idy, ax_idx].sharex(ax[ax_idx, ax_idx])
                plot_kde(
                    pooled_samples[var_x][var_idx],
                    pooled_samples[var_y][var_idy],
                    ax=ax[ax_idy, ax_idx],
                    **kde_kwargs,
                )
                if true_vals.get(var_x) is not None:
                    if len(var_idx) > 0:
                        ax[ax_idy, ax_idx].axvline(
                            np.array(true_vals[var_x])[var_idx], **true_kwargs
                        )
                    else:
                        ax[ax_idy, ax_idx].axvline(
                            true_vals[var_x], **true_kwargs
                        )
                if true_vals.get(var_y) is not None:
                    if len(var_idy) > 0:
                        ax[ax_idy, ax_idx].axhline(
                            np.array(true_vals[var_y])[var_idy], **true_kwargs
                        )
                    else:
                        ax[ax_idy, ax_idx].axhline(
                            true_vals[var_y], **true_kwargs
                        )
            if var_x in var_labels:
                var_x_suffix = "".join(
                    [f"$_{x_coord}$" for x_coord in var_idx]
                )
                ax[-1, ax_idx].set_xlabel(var_labels[var_x] + var_x_suffix)

    fig.align_labels()
    fig.subplots_adjust(wspace=0, hspace=0)

    return fig, ax, bins
