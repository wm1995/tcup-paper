from glob import glob
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from tcup_paper.data.io import load_dataset

L = 1023
B = 4

def pool_bins(bins, pooling_factor):
    # Check pooling factor, bin sizes are powers of 2
    assert np.isclose(np.mod(np.log2(pooling_factor), 1), 0), "pooling_factor must be power of 2"
    assert np.isclose(np.mod(np.log2(bins.shape[0]), 1), 0), "len(bins) must be power of 2"

    if pooling_factor == 1:
        return bins
    else:
        new_bins = np.zeros(bins.shape[0] // 2)
        new_bins = bins[:-1:2] + bins[1::2]
        return pool_bins(new_bins, pooling_factor // 2)


if __name__ == "__main__":
    bins = np.zeros(L + 1)
    N = 0
    var_name = "alpha_scaled"

    for filename in glob("results/t_*.nc"):
        # Attempt to load corresponding data file
        data_filename = "data/" + filename.split("/")[1].split(".")[0] + ".json"
        try:
            _, info = load_dataset(data_filename)
        except:
            continue

        # Load mcmc samples
        mcmc = az.from_netcdf(filename)
        samples = mcmc.posterior[var_name].values.flatten()

        # Thin samples to L
        thinning_factor = samples.shape[0] // L
        # print(f"{samples.shape=}, {thinning_factor=}, {az.ess(samples[::thinning_factor][:L])=}")
        samples = samples[::thinning_factor][:L]
        assert samples.shape == (L,), f"{samples.shape=} != {L}"

        # Calculate rank statistic and add to histogram
        rank = (samples > info[var_name]).sum()
        bins[rank] += 1
        N += 1

    pooled_bins = pool_bins(bins, pooling_factor=(L + 1) // B)
    plt.bar(range(B), pooled_bins)
    plt.show()
