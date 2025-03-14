import numpy as np
import scipy.stats as sps


def gen_dataset(seed):
    rng = np.random.default_rng(seed)
    shape = (25,)
    info = {
        "name": r"Laplace distributed",
        "alpha": -1,
        "beta": 0.8,
        "sigma_int": 0.2,
        "sigma_68": 0.2 * sps.laplace.ppf(sps.norm.cdf(1)),
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

    return x, y, dx, dy, info
