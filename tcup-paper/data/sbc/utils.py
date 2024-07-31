import numpy as np
import scipy.stats as sps


def draw_x_true(rng, n_data, dim_x, weights, means, vars):
    n_mix = sps.multinomial(n_data, weights).rvs(random_state=rng).flatten()
    x_true = []
    for component_size, mu, sigma in zip(n_mix, means, vars):
        component = sps.multivariate_normal(mu, sigma).rvs(
            size=component_size, random_state=rng
        )
        x_true.append(component.reshape(component_size, dim_x))
    return np.concatenate(x_true)
