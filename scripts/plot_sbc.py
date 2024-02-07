import argparse
from glob import glob

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from tcup.utils import sigma_68
from tcup_paper.data.io import load_dataset
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
        case "beta_scaled.0":
            latex_var = r"\tilde{\beta}_0"
        case "beta_scaled.1":
            latex_var = r"\tilde{\beta}_1"
        case "sigma_scaled":
            latex_var = r"\tilde{\sigma}"
        case "nu":
            latex_var = r"\nu"
        case "sigma_68":
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
    dataset_type = parser.add_mutually_exclusive_group(required=True)
    dataset_type.add_argument("--t-dist", action="store_true")
    dataset_type.add_argument("--fixed-nu", action="store_true")
    dataset_type.add_argument("--normal", action="store_true")
    dataset_type.add_argument("--outlier", action="store_true")
    dataset_type.add_argument("--gaussian-mix", action="store_true")
    dataset_type.add_argument("--laplace", action="store_true")
    dataset_type.add_argument("--lognormal", action="store_true")
    args = parser.parse_args()

    var_names = [
        "alpha_scaled",
        "beta_scaled.0",
        "beta_scaled.1",
        "sigma_scaled",
        "nu",
        "sigma_68",
    ]
    var_cdfs = [
        sps.norm(scale=3).cdf,
        sps.cauchy().cdf,
        sps.cauchy().cdf,
        sps.gamma(a=2, scale=1 / 2).cdf,
        sps.invgamma(a=3, scale=10).cdf,
        None,
    ]
    if not (args.tcup and args.t_dist):
        var_names.remove("nu")
        var_names.remove("sigma_68")

    if args.tcup:
        model = "tcup"
    elif args.ncup:
        model = "ncup"
    elif args.fixed:
        model = "fixed3"

    if args.t_dist:
        dataset = "t"
    elif args.fixed_nu:
        dataset = "fixed"
    elif args.normal:
        dataset = "normal"
    elif args.outlier:
        dataset = "outlier"
    elif args.gaussian_mix:
        dataset = "gaussian_mix"
    elif args.laplace:
        dataset = "laplace"
    elif args.lognormal:
        dataset = "lognormal"

    results_path = f"results/sbc/{model}/{dataset}/"
    data_path = f"data/sbc/{dataset}/"
    plots_path = f"plots/sbc/{model}/{dataset}/"

    bins = np.zeros((len(var_names), L + 1))
    ranks = {var_name: [] for var_name in var_names}
    N = 0

    for filename in glob(results_path + "*.nc"):
        # Attempt to load corresponding data file
        data_filename = (
            data_path + filename.split("/")[-1].split(".")[0] + ".json"
        )
        try:
            _, info = load_dataset(data_filename)
        except:
            continue

        # Load mcmc samples
        mcmc = az.from_netcdf(filename)
        # Use thinning given in Section 5.1 of Talts et al. 2018
        min_ess = (
            az.ess(
                mcmc,
                var_names=[var_name.split(".")[0] for var_name in var_names],
            )
            .to_array()
            .min()
        )
        thinning_factor = int(np.ceil(mcmc.posterior.num_samples / min_ess))
        if mcmc.posterior.num_samples // thinning_factor < L:
            # In this case, the thinning prescribed by Talts doesn't leave
            # enough samples to actually do SBC, so use this prescription that's
            # guaranteed to work
            thinning_factor = mcmc.posterior.num_samples // L

        for var_name, curr_bins in zip(var_names, bins):
            if "." in var_name:
                raw_var_name, var_dim = var_name.split(".")
                var_dim = int(var_dim)
                samples = (
                    mcmc.posterior[raw_var_name]
                    .values[:, :, var_dim]
                    .flatten()
                )
                dataset_value = info[raw_var_name][var_dim]
            elif var_name == "sigma_68":
                nu = mcmc.posterior["nu"].values.flatten()
                sigma = mcmc.posterior["sigma_scaled"].values.flatten()
                samples = sigma_68(nu) * sigma
                dataset_value = sigma_68(info["nu"]) * info["sigma_scaled"]
            else:
                samples = mcmc.posterior[var_name].values.flatten()
                dataset_value = info[var_name]

            # Thin samples to L
            samples = samples[::thinning_factor][:L]
            assert samples.shape == (L,), f"{samples.shape=} != {L}"

            # # For nu, invert to get 1/nu
            # if var_name == "nu":
            #     samples = 1 / samples
            #     dataset_value = 1 / dataset_value

            # Calculate rank statistic and add to histogram
            rank = (samples < dataset_value).sum()
            curr_bins[rank] += 1

            # Save rank and dataset value to appropriate value
            ranks[var_name].append((dataset_value, rank, np.median(samples)))

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
            f"{confidence_level * 100:.0f}% confidence interval: [{lower}, {upper}]"
        )

    apply_matplotlib_style()
    for var_name, curr_bins, cdf in zip(var_names, bins, var_cdfs):
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
        plt.scatter(data[:, 0], data[:, 2] - data[:, 0], c=data[:, 1] / L)
        cbar = plt.colorbar()
        latex_var = get_latex_var(var_name)
        plt.xlabel(rf"Dataset ${latex_var}$")
        plt.ylabel(rf"Residual $\hat{{{latex_var}}} - {latex_var}$")
        cbar.set_label(labeller(var_name))
        plt.savefig(f"{plots_path}{var_name}_residuals.pdf")
        plt.close()

        # Produce a plot of CDF(true_val) vs CDF(median_value)
        data = np.array(ranks[var_name])
        plt.scatter(cdf(data[:, 0]), cdf(data[:, 2]), c=data[:, 1] / L)
        cbar = plt.colorbar()
        latex_var = get_latex_var(var_name)
        plt.xlabel(rf"CDF of dataset ${latex_var}$")
        plt.ylabel(rf"CDF of median $\hat{{{latex_var}}}$")
        plt.text(
            0.02,
            0.98,
            f"R = {sps.pearsonr(cdf(data[:, 0]), cdf(data[:, 2])).statistic:.3f}",
            horizontalalignment="left",
            verticalalignment="top",
        )
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        cbar.set_label(labeller(var_name))
        plt.savefig(f"{plots_path}{var_name}_correlation.pdf")
        plt.close()
