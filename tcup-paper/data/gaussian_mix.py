import numpy as np
import scipy.stats as sps


def gen_dataset(seed):
    rng = np.random.default_rng(seed)
    shape = (200, 2)
    info = {
        "name": r"Gaussian mixture",
        "alpha": 2,
        "beta": [3, -1],
        "sigma_int": 0.4,
        "outlier_frac": 0.1,
    }
    outlier_mask = (
        sps.bernoulli(info["outlier_frac"])
        .rvs(size=shape[0], random_state=rng)
        .astype(bool)
    )
    info["outlier_mask"] = outlier_mask.tolist()

    x_true = np.concatenate(
        [
            sps.multivariate_normal.rvs([-3, 2], [[0.5, -1], [-1, 4]], 140),
            sps.multivariate_normal.rvs([-1, -1], [[1, 0.2], [0.2, 0.8]], 60),
        ]
    )
    int_scatter = sps.norm(scale=info["sigma_int"]).rvs(
        size=shape[0], random_state=rng
    )
    epsilon = np.where(outlier_mask, 10 * int_scatter, int_scatter)
    y_true = info["alpha"] + np.dot(x_true, info["beta"]) + epsilon
    cov_x = sps.wishart.rvs(3, np.diag([0.1, 0.1]), size=shape[0])
    eps_x = np.array(
        [sps.multivariate_normal.rvs([0, 0], cov) for cov in cov_x]
    )
    x = x_true + eps_x
    dy = 10 ** sps.norm(-1, 0.1).rvs(size=shape[0], random_state=rng)
    y = sps.norm(loc=y_true, scale=dy).rvs(random_state=rng)

    return x, y, cov_x, dy, info
