import numpy as np
import scipy.stats as sps


def gen_dataset(seed):
    rng = np.random.default_rng(seed)
    shape = (20,)
    info = {
        "name": r"\textit{t}-distributed ($\nu = 3$)",
        "alpha": 3,
        "beta": 2,
        "sigma_int": 0.1,
        "nu": 3,
    }
    x_true = sps.norm(2, 2).rvs(size=shape, random_state=rng)
    epsilon = sps.t(info["nu"], scale=info["sigma_int"]).rvs(
        size=shape[0], random_state=rng
    )
    y_true = info["alpha"] + np.dot(x_true, info["beta"]) + epsilon
    dx = 10 ** sps.norm(-1, 0.1).rvs(size=shape, random_state=rng)
    dy = 10 ** sps.norm(-0.7, 0.1).rvs(size=shape[0], random_state=rng)
    x = sps.t(info["nu"], loc=x_true, scale=dx).rvs(random_state=rng)
    y = sps.t(info["nu"], loc=y_true, scale=dy).rvs(random_state=rng)

    return x, y, dx, dy, info
