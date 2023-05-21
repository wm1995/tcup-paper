import numpy as np
import scipy.stats as sps


def gen_dataset(seed):
    rng = np.random.default_rng(seed)
    shape = (25,)
    info = {
        "name": r"Lognormally distributed",
        "alpha": 4,
        "beta": 8,
        "sigma_int": 0.2,
    }
    x_true = sps.uniform(0, 10).rvs(size=shape, random_state=rng)
    mu = info["alpha"] + np.dot(x_true, info["beta"])
    epsilon = sps.norm(scale=info["sigma_int"]).rvs(
        size=shape[0], random_state=rng
    )
    log_y_true = np.log10(mu) + epsilon
    y_true = 10**log_y_true
    dx = 10 ** sps.norm(-1, 0.1).rvs(size=shape, random_state=rng)
    dy = 10 ** sps.norm(-1, 0.1).rvs(size=shape[0], random_state=rng)
    x = sps.norm(loc=x_true, scale=dx).rvs(random_state=rng)
    y = sps.norm(loc=y_true, scale=dy).rvs(random_state=rng)

    return x, y, dx, dy, info
