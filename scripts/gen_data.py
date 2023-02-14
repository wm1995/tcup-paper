#!/usr/bin/env python
import json
import numpy as np
import scipy.stats as sps

# Set up run parameters
SEED = 24601


def write_dataset(name, data, params):
    output = {
        "data": data,
        "params": params,
    }
    with open(f"data/{name}.json", "w") as f:
        json.dump(output, f)


def t_dist(rng, x_true, alpha, beta, sigma_int, nu):
    (N, K) = x_true.shape

    y_true = (
        alpha
        + np.dot(x_true, beta)
        + sps.t.rvs(nu, sigma_int, size=(N,), random_state=rng)
    )

    return y_true


def lin_rel(rng, x_true, alpha, beta, sigma_int):
    (N, K) = x_true.shape

    y_true = (
        alpha
        + np.dot(x_true, beta)
        + sps.norm.rvs(0, sigma_int, size=(N,), random_state=rng)
    )

    return y_true


def simple_x_obs(rng, x_true, err):
    (N, K) = x_true.shape

    dx = np.array([np.identity(K) for _ in range(N)]) * err
    x_err = np.array(
        [
            sps.multivariate_normal.rvs(np.zeros(K), cov, random_state=rng)
            for cov in dx
        ]
    ).reshape(N, K)
    x = x_true + x_err

    return x, dx


def simple_y_obs(rng, y_true, err):
    (N,) = y_true.shape

    dy = np.ones(y_true.shape) * err
    y = sps.norm.rvs(y_true, dy, random_state=rng)

    return y, dy


def no_outlier(rng, y_true):
    return y_true, np.ones(y_true.shape, dtype=bool)


def simple_outlier(rng, y_true, outlier):
    (N,) = y_true.shape

    # Set up outlier mask
    mask = np.ones(N, dtype=bool)
    mask[-2] = False

    y_true[~mask] -= outlier

    return y_true, mask


def normal_outlier(rng, y_true, sigma_outlier, outlier_frac):
    (N,) = y_true.shape

    N_outlier = rng.binomial(N, outlier_frac)

    # Set up outlier mask
    mask = np.ones(N, dtype=bool)
    inds = rng.choice(np.arange(N), N_outlier, replace=False)
    mask[inds] = False

    y_true[~mask] += sps.norm.rvs(
        0, sigma_outlier, size=(N_outlier,), random_state=rng
    )

    return y_true, mask


def gen_data(
    seed,
    x_true,
    y_true_fn,
    y_true_params,
    x_obs_fn,
    x_obs_params,
    y_obs_fn,
    y_obs_params,
    outlier_fn,
    outlier_params,
):
    # Set up RNG
    rng = np.random.default_rng(seed)

    # Generate true y values
    y_true = y_true_fn(rng, x_true, **y_true_params)

    # Add outliers
    y_true, outlier_mask = outlier_fn(rng, y_true, **outlier_params)

    # Generate observed values
    x, dx = x_obs_fn(rng, x_true, **x_obs_params)
    y, dy = y_obs_fn(rng, y_true, **y_obs_params)

    # Return data as dict
    data = {
        "x": x.tolist(),
        "dx": dx.tolist(),
        "y": y.tolist(),
        "dy": dy.tolist(),
    }
    params = {
        "y_true_params": y_true_params,
        "outliers": (~outlier_mask).tolist(),
        "x_obs_params": x_obs_params,
        "y_obs_params": y_obs_params,
    }
    return data, params


if __name__ == "__main__":
    # Create datasets
    # Dataset 1: linear relationship, dim x = 1, 1 outlier
    data, params = gen_data(
        seed=SEED,
        x_true=np.linspace(0, 10, 12)[:, np.newaxis],
        y_true_fn=lin_rel,
        y_true_params={
            "alpha": 3,
            "beta": [2],
            "sigma_int": 0.1,
        },
        x_obs_fn=simple_x_obs,
        x_obs_params={"err": 0.2},
        y_obs_fn=simple_y_obs,
        y_obs_params={"err": 0.2},
        outlier_fn=simple_outlier,
        outlier_params={"outlier": 10},
    )
    write_dataset("linear_1D", data, params)

    # Dataset 2: linear relationship, dim x = 2, 1 outlier
    data, params = gen_data(
        seed=SEED,
        x_true=np.random.default_rng(SEED).uniform(-1, 1, 200).reshape(-1, 2),
        y_true_fn=lin_rel,
        y_true_params={
            "alpha": 3,
            "beta": [2, 1],
            "sigma_int": 0.1,
        },
        x_obs_fn=simple_x_obs,
        x_obs_params={"err": 0.2},
        y_obs_fn=simple_y_obs,
        y_obs_params={"err": 0.2},
        outlier_fn=simple_outlier,
        outlier_params={"outlier": 1.5},
    )
    write_dataset("linear_2D", data, params)

    # Dataset 3: linear relationship, dim x = 3, 1 outlier
    data, params = gen_data(
        seed=SEED,
        x_true=np.random.default_rng(SEED).uniform(-1, 1, 90).reshape(-1, 3),
        y_true_fn=lin_rel,
        y_true_params={
            "alpha": 3,
            "beta": [2, 1, 3],
            "sigma_int": 0.1,
        },
        x_obs_fn=simple_x_obs,
        x_obs_params={"err": 0.2},
        y_obs_fn=simple_y_obs,
        y_obs_params={"err": 0.2},
        outlier_fn=simple_outlier,
        outlier_params={"outlier": 1.5},
    )
    write_dataset("linear_3D", data, params)

    # Dataset 4: t-distribution data, dim x = 1, 1 outlier
    data, params = gen_data(
        seed=SEED,
        x_true=np.linspace(-5, 5, 40)[:, np.newaxis],
        y_true_fn=t_dist,
        y_true_params={
            "alpha": -1,
            "beta": [0.5],
            "sigma_int": 0.02,
            "nu": 4,
        },
        x_obs_fn=simple_x_obs,
        x_obs_params={"err": 0.1},
        y_obs_fn=simple_y_obs,
        y_obs_params={"err": 0.1},
        outlier_fn=no_outlier,
        outlier_params={},
    )
    write_dataset("t_dist_1D", data, params)

    # Dataset 5: Gaussian mixture, dim x = 1
    data, params = gen_data(
        seed=SEED,
        x_true=np.linspace(-3, 6, 100)[:, np.newaxis],
        y_true_fn=lin_rel,
        y_true_params={
            "alpha": -1,
            "beta": [0.5],
            "sigma_int": 0.02,
        },
        x_obs_fn=simple_x_obs,
        x_obs_params={"err": 0.1},
        y_obs_fn=simple_y_obs,
        y_obs_params={"err": 0.1},
        outlier_fn=normal_outlier,
        outlier_params={
            "sigma_outlier": 0.2,
            "outlier_frac": 0.15,
        },
    )
    write_dataset("gaussian_mixture_1D", data, params)
