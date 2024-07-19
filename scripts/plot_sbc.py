import argparse
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import xarray as xr
from tcup_paper.data.io import load_dataset
from tcup_paper.model import prior
from tcup_paper.plot.style import apply_matplotlib_style

L = 1023
B = 16


def pool_bins(bins, pooling_factor):
    # Check pooling factor, bin sizes are powers of 2
    assert np.isclose(
        np.mod(np.log2(pooling_factor), 1), 0
    ), "pooling_factor must be power of 2"
    assert np.isclose(
        np.mod(np.log2(bins.shape[0]), 1), 0
    ), "len(bins) must be power of 2"

    if pooling_factor == 1:
        return bins
    else:
        new_bins = np.zeros(bins.shape[0] // 2)
        new_bins = bins[:-1:2] + bins[1::2]
        return pool_bins(new_bins, pooling_factor // 2)


def get_latex_var(var_name):
    match var_name:
        case "alpha_scaled":
            latex_var = r"\tilde{\alpha}"
        case "beta_scaled":
            latex_var = r"\tilde{\beta}"
        case "beta_scaled.0":
            latex_var = r"\tilde{\beta}_0"
        case "beta_scaled.1":
            latex_var = r"\tilde{\beta}_1"
        case "sigma_scaled":
            latex_var = r"\tilde{\sigma}"
        case "nu":
            latex_var = r"\nu"
        case "sigma_68_scaled":
            latex_var = r"\tilde{\sigma}_{68}"

    return latex_var


def labeller(var_name):
    latex_var = get_latex_var(var_name)
    return rf"Normalised rank statistic $r\left({latex_var}\right)$"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_type = parser.add_mutually_exclusive_group(required=True)
    model_type.add_argument("--tcup", action="store_true")
    model_type.add_argument("--fixed", action="store_true")
    model_type.add_argument("--ncup", action="store_true")
    model_type.add_argument("--tobs", action="store_true")
    dataset_type = parser.add_mutually_exclusive_group(required=True)
    dataset_type.add_argument("--t-dist", action="store_true")
    dataset_type.add_argument("--fixed-nu", action="store_true")
    dataset_type.add_argument("--normal", action="store_true")
    dataset_type.add_argument("--outlier", type=float)
    dataset_type.add_argument("--cauchy-mix", action="store_true")
    dataset_type.add_argument("--gaussian-mix", action="store_true")
    dataset_type.add_argument("--laplace", action="store_true")
    dataset_type.add_argument("--lognormal", action="store_true")
    dataset_type.add_argument("--t-obs", action="store_true")
    args = parser.parse_args()

    var_names = None

    if args.tcup:
        model = "tcup"
    elif args.ncup:
        model = "ncup"
    elif args.fixed:
        model = "fixed3"
    elif args.tobs:
        model = "tobs"

    if args.t_dist:
        dataset = "t"
    elif args.fixed_nu:
        dataset = "fixed"
    elif args.normal:
        dataset = "normal"
    elif args.outlier:
        dataset = f"outlier{int(args.outlier)}"
    elif args.cauchy_mix:
        dataset = "cauchy_mix"
    elif args.gaussian_mix:
        dataset = "gaussian_mix"
    elif args.laplace:
        dataset = "laplace"
    elif args.lognormal:
        dataset = "lognormal"
    elif args.t_obs:
        dataset = "tobs"

    results_path = f"results/sbc/{model}/{dataset}/"
    data_path = f"data/sbc/{dataset}/"
    plots_path = f"plots/sbc/{model}/{dataset}/"

    N = 0

    for filename in glob(results_path + "*.nc"):
        # Attempt to load corresponding data file
        data_filename = (
            data_path + filename.split("/")[-1].split(".")[0] + ".json"
        )
        try:
            _, info = load_dataset(data_filename)
        except FileNotFoundError:
            continue

        sbc_data = xr.load_dataset(filename)

        if var_names is None:
            var_names = []
            var_cdfs = []
            for x in sbc_data.keys():
                if "rank_" in x:
                    continue
                elif "true_" in x:
                    continue
                elif "post_pred_" in x:
                    continue
                elif x == "beta_scaled":
                    for idx in range(sbc_data.sizes["beta_scaled_dim_0"]):
                        var_names.append(f"beta_scaled.{idx}")
                        var_cdfs.append(prior.beta_prior.cdf)
                else:
                    var_names.append(x)
                    if x == "alpha_scaled":
                        var_cdfs.append(prior.alpha_prior.cdf)
                    elif x == "sigma_68_scaled":
                        var_cdfs.append(prior.sigma_68_prior.cdf)
                    elif x == "nu":
                        var_cdfs.append(prior.nu_prior.cdf)

            ranks = {var_name: [] for var_name in var_names}
            bins = np.zeros((len(var_names), L + 1))

        for var_name, curr_bins in zip(var_names, bins):
            if "beta_scaled" in var_name:
                idx = int(var_name.split(".")[-1])
                # Save rank and dataset value to appropriate value
                ranks[var_name].append(
                    (
                        sbc_data["true_beta_scaled"].values[idx],
                        sbc_data["rank_beta_scaled"].values[idx],
                        sbc_data["beta_scaled"].median(axis=-1)[idx],
                    )
                )
                bin_idx = int(sbc_data["rank_beta_scaled"].values[idx] * L)
                curr_bins[bin_idx] += 1
            else:
                # Save rank and dataset value to appropriate value
                ranks[var_name].append(
                    (
                        sbc_data[f"true_{var_name}"].values[()],
                        sbc_data[f"rank_{var_name}"].values[()],
                        sbc_data[var_name].median(axis=-1)[()],
                    )
                )
                bin_idx = int(sbc_data[f"rank_{var_name}"].values[()] * L)
                curr_bins[bin_idx] += 1

        # Increment number of datasets
        N += 1

    confidence_levels = [1 - 1 / B, 1 - 1 / (len(var_names) * B)]
    confidence_intervals = []
    for confidence_level in confidence_levels:
        cdf_vals = [0.5 - confidence_level / 2, 0.5 + confidence_level / 2]
        lower, upper = sps.binom.ppf(cdf_vals, N, 1 / B)
        confidence_intervals.append((lower, upper))
        print(f"{N=}, {B=}, {N / B=:.2f}")
        print(
            f"{confidence_level * 100:.0f}% confidence interval: "
            f"[{lower}, {upper}]"
        )

    apply_matplotlib_style()
    fig, ax = plt.subplots(1, len(var_names), sharey=True, figsize=(8.5, 2.5))
    for idx, (var_name, curr_bins) in enumerate(zip(var_names, bins)):
        pooled_bins = pool_bins(curr_bins, pooling_factor=(L + 1) // B)
        for lower, upper in confidence_intervals:
            ax[idx].fill_between(
                [-0.05, 1.05], lower, upper, alpha=0.1, color="k"
            )
        edges = np.linspace(0, 1 - 1 / B, B)
        ax[idx].bar(edges, pooled_bins, width=1 / B, align="edge")
        ax[idx].hlines(
            [N / B],
            xmin=-0.05,
            xmax=1.05,
            colors="k",
            linestyles="dashed",
        )
        ax[idx].set_xlabel(rf"$r({get_latex_var(var_name)})$")
        ax[idx].set_xlim(-0.02, 1.02)
        ax[idx].set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{plots_path}sbc.pdf")
    plt.close()

    for var_name, curr_bins, cdf in zip(var_names, bins, var_cdfs):
        # Check CDF of rank is uniform
        ks_test = sps.kstest([x[1] for x in ranks[var_name]], sps.uniform.cdf)
        print(f"{var_name} has KS p={ks_test.pvalue:03f}")

        pooled_bins = pool_bins(curr_bins, pooling_factor=(L + 1) // B)
        for lower, upper in confidence_intervals:
            plt.fill_between([-0.05, 1.05], lower, upper, alpha=0.1, color="k")
        edges = np.linspace(0, 1 - 1 / B, B)
        plt.bar(edges, pooled_bins, width=1 / B, align="edge")
        plt.hlines(
            [N / B],
            xmin=-0.05,
            xmax=1.05,
            colors="k",
            linestyles="dashed",
        )
        plt.xlabel(labeller(var_name))
        plt.xlim(-0.02, 1.02)
        plt.yticks([])
        plt.savefig(f"{plots_path}{var_name}.pdf")
        plt.close()

        if cdf is None:
            continue

        # Produce a residual plot to visualise which regions are problematic
        data = np.array(ranks[var_name])
        plt.scatter(data[:, 0], data[:, 2] - data[:, 0], c=data[:, 1])
        cbar = plt.colorbar()
        latex_var = get_latex_var(var_name)
        plt.xlabel(rf"Dataset ${latex_var}$")
        plt.ylabel(rf"Residual $\hat{{{latex_var}}} - {latex_var}$")
        cbar.set_label(labeller(var_name))
        plt.savefig(f"{plots_path}{var_name}_residuals.pdf")
        plt.close()

        # Produce a plot of CDF(true_val) vs CDF(median_value)
        data = np.array(ranks[var_name])
        plt.scatter(cdf(data[:, 0]), cdf(data[:, 2]), c=data[:, 1])
        cbar = plt.colorbar()
        latex_var = get_latex_var(var_name)
        plt.xlabel(rf"CDF of dataset ${latex_var}$")
        plt.ylabel(rf"CDF of median $\hat{{{latex_var}}}$")
        pearsonr = sps.pearsonr(cdf(data[:, 0]), cdf(data[:, 2])).statistic
        plt.text(
            0.02,
            0.98,
            f"R = {pearsonr:.3f}",
            horizontalalignment="left",
            verticalalignment="top",
        )
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        cbar.set_label(labeller(var_name))
        plt.savefig(f"{plots_path}{var_name}_correlation.pdf")
        plt.close()
