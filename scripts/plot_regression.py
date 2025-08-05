import argparse
import json

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from tcup_paper.plot import style

SEED = 2023


def load_dataset(filename):
    with open(filename, "r") as f:
        dataset = json.load(f)

    data = {key: np.array(val) for key, val in dataset["data"].items()}
    info = dataset.get("info", {})
    return data, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tcup-file", required=True)
    parser.add_argument("--ncup-file", required=True)
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("min", "max"),
        required=True,
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("min", "max"),
        required=False,
    )
    parser.add_argument("--xlabel", type=str, default=r"Observed $\hat{x}$")
    parser.add_argument("--ylabel", type=str, default=r"Observed $\hat{y}$")
    parser.add_argument("--no-errorbars", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Set matplotlib style
    style.apply_matplotlib_style()

    # Load dataset
    data, info = load_dataset(args.dataset)

    # Load mcmc data
    t_mcmc = az.from_netcdf(args.tcup_file)
    n_mcmc = az.from_netcdf(args.ncup_file)

    colors = []
    if "linmix" in args.ncup_file:
        colors.append("cornflowerblue")
    else:
        colors.append("blue")
    colors.append("red")

    x_axis = np.linspace(*args.xlim, 200)

    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(1, 2, figsize=(7.06, 3.57), sharey=True)
    for idx, (ax_i, mcmc, color) in enumerate(
        zip(ax, [n_mcmc, t_mcmc], colors)
    ):
        inds = rng.choice(
            mcmc["posterior"].sizes["chain"] * mcmc["posterior"].sizes["draw"],
            size=100,
        )
        ax_i.plot(
            x_axis,
            [
                mcmc["posterior"]["alpha"].values.flatten()[inds]
                + mcmc["posterior"]["beta"].values.flatten()[inds] * x_val
                for x_val in x_axis
            ],
            color=color,
            alpha=0.1 if color == "cornflowerblue" else 0.05,
        )
        try:
            ax_i.plot(
                x_axis,
                info["alpha"] + info["beta"] * x_axis,
                color="k",
                linestyle="dashed",
            )
        except KeyError:
            pass
        if args.no_errorbars:
            ax_i.plot(
                data["x"],
                data["y"],
                "k+",
            )
        else:
            ax_i.errorbar(
                data["x"],
                data["y"],
                data["dy"],
                data["dx"],
                "k+",
            )
        ax_i.set_xlim(args.xlim)
        if args.ylim:
            ax_i.set_ylim(args.ylim)
        if args.xlabel:
            ax_i.set_xlabel(args.xlabel)
        if args.ylabel and idx == 0:
            ax_i.set_ylabel(args.ylabel)

    plt.tight_layout()
    plt.savefig(args.output)
