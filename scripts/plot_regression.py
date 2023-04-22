import argparse

import json
import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SEED = 42


def load_dataset(filename):
    with open(filename, "r") as f:
        dataset = json.load(f)

    data = {key: np.array(val) for key, val in dataset["data"].items()}
    params = dataset["params"]
    params["outliers"] = np.array(params["outliers"])
    return data, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prior", required=True)
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
    data, params = load_dataset(f"data/{args.dataset}.json")

    # Load mcmc data
    t_mcmc = az.from_netcdf(f"results/{args.dataset}_{args.prior}.nc")
    n_mcmc = az.from_netcdf(f"results/{args.dataset}_ncup.nc")

    x_axis = np.linspace(0, 10, 200)

    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(1, 2, figsize=(7.06, 3.57), sharey=True)
    for idx, (ax_i, mcmc) in enumerate(zip(ax, [n_mcmc, t_mcmc])):
        inds = rng.choice(
            mcmc["posterior"].dims["chain"] * mcmc["posterior"].dims["draw"],
            size=100,
        )
        ax_i.plot(
            x_axis,
            [
                mcmc["posterior"]["alpha"].values.flatten()[inds]
                + mcmc["posterior"]["beta"].values.flatten()[inds] * x_val
                for x_val in x_axis
            ],
            color="red" if idx else "blue",
            alpha=0.05,
        )
        ax_i.plot(
            x_axis,
            params["y_true_params"]["alpha"]
            + params["y_true_params"]["beta"][0] * x_axis,
            color="k",
            linestyle="dashed",
        )
        ax_i.errorbar(
            [x_i[0] for x_i in data["x"]],
            data["y"],
            data["dy"],
            [dx_i[0][0] for dx_i in data["dx"]],
            "k+",
        )
        ax_i.set_xlim((0, 10))

    plt.tight_layout()
    plt.savefig(
        f"plots/regression_{args.dataset}_{args.prior}.pdf", backend="pgf"
    )