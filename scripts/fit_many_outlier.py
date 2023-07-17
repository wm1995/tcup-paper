import contextlib
import io
import sys

import numpy as np
from tcup import tcup
import tqdm

from tcup_paper.data.outlier import gen_dataset


@contextlib.contextmanager
def suppress_output():
    save_stderr = sys.stderr
    sys.stderr = io.StringIO()
    yield
    sys.stderr = save_stderr


def gen_data(seed):
    x, y, dx, dy, info = gen_dataset(seed)
    data = {
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
    }

    # Exclude outlier and save
    idx = info["outlier_idx"]
    mask = np.ones(shape=y.shape, dtype=bool)
    mask[idx] = False
    no_outlier_data = {
        "x": x[mask],
        "y": y[mask],
        "dx": dx[mask],
        "dy": dy[mask],
    }

    return data, no_outlier_data


if __name__ == "__main__":
    for seed in tqdm.trange(1000):
        data, no_outlier_data = gen_data(seed)
        with suppress_output():
            mcmc = tcup(data, seed, num_samples=4000)
            n_mcmc = tcup(data, seed, model="ncup", num_samples=4000)
            no_outlier_mcmc = tcup(no_outlier_data, seed, num_samples=4000)
            no_outlier_n_mcmc = tcup(
                no_outlier_data, seed, model="ncup", num_samples=4000
            )
        mcmc.to_netcdf(f"results/repeats/outlier_tcup_{seed}.nc")
        n_mcmc.to_netcdf(f"results/repeats/outlier_ncup_{seed}.nc")
        no_outlier_mcmc.to_netcdf(f"results/repeats/normal_tcup_{seed}.nc")
        no_outlier_n_mcmc.to_netcdf(f"results/repeats/normal_ncup_{seed}.nc")
