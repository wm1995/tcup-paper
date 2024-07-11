import numpy as np
import scipy.stats as sps
from tcup.utils import sigma_68


def gen_dataset(seed):
    rng = np.random.default_rng(seed)
    shape = (20,)
    info = {
        "name": r"\textit{t}-distributed ($\nu = 3$)",
        "alpha": 3,
        "beta": 2,
        "sigma_68": 0.1,
        "nu": 3,
    }
    t_scale = info["sigma_68"] / sigma_68(info["nu"])
    x_true = sps.norm(2, 2).rvs(size=shape, random_state=rng)
    epsilon = sps.t(info["nu"], scale=t_scale).rvs(size=shape[0], random_state=rng)
    y_true = info["alpha"] + np.dot(x_true, info["beta"]) + epsilon
    dx = 10 ** sps.norm(-1, 0.1).rvs(size=shape, random_state=rng)
    dy = 10 ** sps.norm(-0.7, 0.1).rvs(size=shape[0], random_state=rng)
    x = sps.norm(loc=x_true, scale=dx).rvs(random_state=rng)
    y = sps.norm(loc=y_true, scale=dy).rvs(random_state=rng)

    return x, y, dx, dy, info
