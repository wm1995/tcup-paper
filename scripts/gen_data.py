#!/usr/bin/env python
import json
import numpy as np
import scipy.stats as sps

# Set up run parameters
SEED = 24601


def write_dataset(name, data, info):
    output = {
        "data": data,
        "info": info,
    }
    with open(f"data/{name}.json", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    # Dataset 1
    # t-distributed
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
    data = {
        "x": x.tolist(),
        "y": y.tolist(),
        "dx": dx.tolist(),
        "dy": dy.tolist(),
    }
    write_dataset("t", data, info)

    # Dataset 2
    # Normally distributed with single outlier
    shape = (12,)
    info = {
        "name": r"Normally distributed with outlier",
        "alpha": 3,
        "beta": 2,
        "sigma_int": 0.1,
        "outlier_idx": 10,
    }
    x_true = sps.norm(-3, 0.5).rvs(size=shape, random_state=rng)
    epsilon = sps.norm(scale=info["sigma_int"]).rvs(
        size=shape[0], random_state=rng
    )
    y_true = info["alpha"] + np.dot(x_true, info["beta"]) + epsilon
    y_true[info["outlier_idx"]] -= 10
    dx = 10 ** sps.norm(-1, 0.1).rvs(size=shape, random_state=rng)
    dy = 10 ** sps.norm(-1, 0.1).rvs(size=shape[0], random_state=rng)
    x = sps.norm(loc=x_true, scale=dx).rvs(random_state=rng)
    y = sps.norm(loc=y_true, scale=dy).rvs(random_state=rng)
    data = {
        "x": x.tolist(),
        "y": y.tolist(),
        "dx": dx.tolist(),
        "dy": dy.tolist(),
    }
    write_dataset("normal", data, info)

    # Dataset 3
    # Spike-and-slab-type Gaussian mixture
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
    data = {
        "x": x.tolist(),
        "y": y.tolist(),
        "cov_x": cov_x.tolist(),
        "dy": dy.tolist(),
    }
    write_dataset("gaussian_mix", data, info)

    # Dataset 4
    # Laplace distributed
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
        "x": x.tolist(),
        "y": y.tolist(),
        "dx": dx.tolist(),
        "dy": dy.tolist(),
    }
    write_dataset("laplace", data, info)

    # Dataset 5
    # Lognormal
    shape = (25,)
    info = {
        "name": r"Lognormally distributed",
        "alpha": 4,
        "beta": 8,
        "sigma_int": 0.2,
    }
    x_true = sps.uniform(-5, 5).rvs(size=shape, random_state=rng)
    epsilon = 10 ** sps.norm(loc=np.log10(info["sigma_int"]), scale=0.1).rvs(
        size=shape[0], random_state=rng
    )
    epsilon = sps.laplace(scale=info["sigma_int"]).rvs(
        size=shape[0], random_state=rng
    )
    y_true = info["alpha"] + np.dot(x_true, info["beta"]) + epsilon
    dx = 10 ** sps.norm(-1, 0.1).rvs(size=shape, random_state=rng)
    dy = 10 ** sps.norm(-1, 0.1).rvs(size=shape[0], random_state=rng)
    x = sps.norm(loc=x_true, scale=dx).rvs(random_state=rng)
    y = sps.norm(loc=y_true, scale=dy).rvs(random_state=rng)
    data = {
        "x": x.tolist(),
        "y": y.tolist(),
        "dx": dx.tolist(),
        "dy": dy.tolist(),
    }
    write_dataset("lognormal", data, info)
