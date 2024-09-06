import argparse
import json

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
    parser.add_argument("--no-errorbars", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Set matplotlib style
    preamble = r"""
    \usepackage{unicode-math}
    \setmainfont{XITS-Regular.otf}
    \setmathfont{XITSMath-Regular.otf}
    """
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["pgf.preamble"] = preamble
    mpl.rcParams["pgf.rcfonts"] = False
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"

    # Load dataset
    data, info = load_dataset(args.dataset)

    # Load mcmc data
    t_mcmc = az.from_netcdf(args.tcup_file)
    n_mcmc = az.from_netcdf(args.ncup_file)

    x_axis = np.linspace(*args.xlim, 200)

    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(1, 2, figsize=(7.06, 3.57), sharey=True)
    for idx, (ax_i, mcmc) in enumerate(zip(ax, [n_mcmc, t_mcmc])):
        inds = rng.choice(
            mcmc["posterior"].sizes["chain"] * mcmc["posterior"].sizes["draw"],
            size=100,
        )
        ax_i.plot(
            x_axis,
            [
                mcmc["posterior"]["alpha"].values.flatten()[inds]
                + mcmc["posterior"]["beta"].values.flatten()[inds]
                * x_val
                for x_val in x_axis
            ],
            color="red" if idx else "blue",
            alpha=0.05,
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

    plt.tight_layout()
    plt.savefig(args.output)
