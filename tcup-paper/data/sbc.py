import numpy as np
import scipy.stats as sps

def draw_x_true(rng, N, D, K, theta_mix, mu_mix, sigma_mix):
    n_mix = sps.multinomial(N, theta_mix).rvs(random_state=rng).flatten()
    x_true = []
    for component_size, mu, sigma in zip(n_mix, mu_mix, sigma_mix):
        component = sps.multivariate_normal(mu, sigma).rvs(size=component_size, random_state=rng)
        x_true.append(component.reshape(component_size, D))
    return np.concatenate(x_true)

def draw_params_from_prior(rng, dim_x=1):
    alpha_scaled = sps.norm(scale=3).rvs(random_state=rng)
    beta_scaled = sps.norm(scale=3).rvs(size=dim_x, random_state=rng)
    gamma_scaled = sps.gamma(a=2, scale=1/2).rvs(random_state=rng)
    nu = sps.invgamma(a=3, scale=10).rvs(random_state=rng)
    return alpha_scaled, beta_scaled, gamma_scaled, nu

def gen_dataset(seed, x_true_params, dist_params):
    rng = np.random.default_rng(seed)
    N = x_true_params["N"]
    dim_x = x_true_params["D"]
    alpha_scaled, beta_scaled, sigma_scaled, nu = draw_params_from_prior(rng, dim_x)
    x_true = draw_x_true(rng, **x_true_params)

    # Generate latent y
    match dist_params:
        case {
            "name": "fixed",
            "nu": fixed_nu,
        }:
            nu = fixed_nu
            epsilon = sps.t(nu, scale=sigma_scaled).rvs(
                size=N, random_state=rng
            )
            y_true = alpha_scaled + np.dot(x_true, beta_scaled) + epsilon
        case {
            "name": "t",
        }:
            epsilon = sps.t(nu, scale=sigma_scaled).rvs(
                size=N, random_state=rng
            )
            y_true = alpha_scaled + np.dot(x_true, beta_scaled) + epsilon
        case {
            "name": "normal",
        }:
            epsilon = sps.norm(scale=sigma_scaled).rvs(
                size=N, random_state=rng
            )
            y_true = alpha_scaled + np.dot(x_true, beta_scaled) + epsilon
        case {
            "name": "outlier",
            "outlier_idx": outlier_idx,
        }:
            x_true = np.sort(x_true)
            epsilon = sps.norm(scale=sigma_scaled).rvs(
                size=N, random_state=rng
            )
            epsilon[outlier_idx] -= 10 * sigma_scaled
            y_true = alpha_scaled + np.dot(x_true, beta_scaled) + epsilon
        case {
            "name": "gaussian_mix",
            "outlier_prob": outlier_prob,
        }:
            outlier_mask = (
                sps.bernoulli(outlier_prob)
                .rvs(size=N, random_state=rng)
                .astype(bool)
            )

            int_scatter = sps.norm(scale=sigma_scaled).rvs(
                size=N, random_state=rng
            )
            epsilon = np.where(outlier_mask, 10 * int_scatter, int_scatter)
            y_true = alpha_scaled + np.dot(x_true, beta_scaled) + epsilon
        case {
            "name": "laplace",
        }:
            epsilon = sps.laplace(scale=sigma_scaled).rvs(
                size=N, random_state=rng
            )
            y_true = alpha_scaled + np.dot(x_true, beta_scaled) + epsilon
        case {
            "name": "lognormal",
        }:
            mu = alpha_scaled + np.dot(x_true, beta_scaled)
            epsilon = sps.norm(scale=0.1 * sigma_scaled).rvs(
                size=N, random_state=rng
            )
            y_true = mu + 10**epsilon
        case _:
            raise NotImplementedError

    info = {
        "true_x": x_true.tolist(),
        "true_y": y_true.tolist(),
        "alpha_scaled": alpha_scaled,
        "beta_scaled": beta_scaled.tolist(),
        "sigma_scaled": sigma_scaled,
        "nu": nu,
    }

    if dist_params["name"] == "gaussian_mix":
        info["outlier_mask"] = outlier_mask.tolist()

    # Observe data
    # Generate observational errors
    cov_x_scaled = sps.wishart.rvs(dim_x + 1, np.diag([0.1] * dim_x), size=N, random_state=rng)
    dy_scaled = 10 ** sps.norm(-1, 0.1).rvs(size=N, random_state=rng)

    match dist_params:
        case {
            "name": "t",
            **other_params,
        }:
            eps_x = np.array(
                [sps.multivariate_t.rvs([0] * dim_x, cov, random_state=rng) for cov in cov_x_scaled]
            )
            x_scaled = x_true + eps_x
            y_scaled = sps.t(nu, loc=y_true, scale=dy_scaled).rvs(random_state=rng)
        case _:
            eps_x = np.array(
                [sps.multivariate_normal.rvs([0] * dim_x, cov, random_state=rng) for cov in cov_x_scaled]
            )
            x_scaled = x_true + eps_x
            y_scaled = sps.norm(loc=y_true, scale=dy_scaled).rvs(random_state=rng)

    data = {
        # Data
        "N": N,
        "D": dim_x,
        "y_scaled": y_scaled.tolist(),
        "dy_scaled": dy_scaled.tolist(),
        "x_scaled": x_scaled.tolist(),
        "cov_x_scaled": cov_x_scaled.tolist(),
        # Mixture priors
        "K": x_true_params["K"],
        "theta_mix": x_true_params["theta_mix"],
        "mu_mix": x_true_params["mu_mix"],
        "sigma_mix": x_true_params["sigma_mix"],
    }

    return data, info