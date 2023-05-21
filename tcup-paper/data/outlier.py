import numpy as np
import scipy.stats as sps


def gen_dataset(seed):
    rng = np.random.default_rng(seed)
    shape = (12,)
    info = {
        "name": r"Normally distributed with outlier",
        "alpha": 3,
        "beta": 2,
        "sigma_int": 0.2,
        "outlier_idx": 10,
    }
    x_true = np.sort(sps.norm(5, 3).rvs(size=shape, random_state=rng))
    epsilon = sps.norm(scale=info["sigma_int"]).rvs(
        size=shape[0], random_state=rng
    )
    y_true = info["alpha"] + np.dot(x_true, info["beta"]) + epsilon
    y_true[info["outlier_idx"]] -= 10
    dx = 10 ** sps.norm(-0.5, 0.1).rvs(size=shape, random_state=rng)
    dy = 10 ** sps.norm(-0.3, 0.1).rvs(size=shape[0], random_state=rng)
    x = sps.norm(loc=x_true, scale=dx).rvs(random_state=rng)
    y = sps.norm(loc=y_true, scale=dy).rvs(random_state=rng)

    return x, y, dx, dy, info
