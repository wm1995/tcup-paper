import numpy as np
import scipy.stats as sps
from scipy.optimize import root


def cdf(x, sigma_int, outlier_prob):
    result = (1 - outlier_prob) * sps.norm.cdf(x, scale=sigma_int)
    result += outlier_prob * sps.norm.cdf(x, scale=10 * sigma_int)
    return result


def sigma_68(sigma_int, outlier_prob):
    # Define a function whose root gives sigma_68
    def f(x):
        # Take advantage of the CDF being symmetric
        return cdf(x, sigma_int, outlier_prob) - sps.norm.cdf(1)

    # Find the root
    result = root(f, sigma_int)

    # If the root was found successfully, return the answer
    assert result.success
    return result.x[0]


def outlier_frac(outlier_prob):
    # This is a symmetric distribution, so outlier fraction is this
    # (no need to scale x and scale by sigma_int because it would cancel)
    return 2 * cdf(-3, 1, outlier_prob)


def gen_dataset(seed):
    rng = np.random.default_rng(seed)
    shape = (200, 2)
    info = {
        "name": r"Gaussian mixture",
        "alpha": 2,
        "beta": [3, -1],
        "sigma_int": 0.4,
        "outlier_probability": 0.1,
    }
    info["sigma_68"] = sigma_68(info["sigma_int"], info["outlier_probability"])
    info["outlier_frac"] = outlier_frac(info["outlier_probability"])

    outlier_mask = (
        sps.bernoulli(info["outlier_probability"])
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
