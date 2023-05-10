import contextlib
import io
import sys

import numpy as np
import scipy.stats as sps
from tcup import tcup
import tqdm


@contextlib.contextmanager
def suppress_output():
    save_stderr = sys.stderr
    sys.stderr = io.StringIO()
    yield
    sys.stderr = save_stderr


def gen_data(seed):
    rng = np.random.default_rng(seed)
    shape = (25,)
    info = {
        "name": r"Laplace distributed",
        "alpha": -1,
        "beta": 0.8,
        "sigma_int": 0.2,
    }
    x_true = sps.uniform(-5, 5).rvs(size=shape, random_state=rng)
    epsilon = sps.laplace(scale=info["sigma_int"]).rvs(
        size=shape[0], random_state=rng
    )
    y_true = info["alpha"] + np.dot(x_true, info["beta"]) + epsilon
    dx = 10 ** sps.norm(-1, 0.1).rvs(size=shape, random_state=rng)
    dy = 10 ** sps.norm(-1, 0.1).rvs(size=shape[0], random_state=rng)
    x = sps.norm(loc=x_true, scale=dx).rvs(random_state=rng)
    y = sps.norm(loc=y_true, scale=dy).rvs(random_state=rng)
    data = {
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
    }
    return data


if __name__ == "__main__":
    for seed in tqdm.trange(1000):
        data = gen_data(seed)
        with suppress_output():
            mcmc = tcup(data, seed, num_samples=4000)
        mcmc.to_netcdf(f"results/repeats/laplace_tcup_{seed}.nc")
