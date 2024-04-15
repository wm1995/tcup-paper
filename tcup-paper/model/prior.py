import scipy.stats as sps

alpha_prior = sps.norm(scale=3)
beta_prior = sps.norm(scale=3)
sigma_68_prior = sps.gamma(a=2, scale=1 / 4)
nu_prior = sps.invgamma(a=4, scale=15)

def draw_params_from_prior(rng, dim_x=1):
    alpha_scaled = alpha_prior.rvs(random_state=rng)
    beta_scaled = beta_prior.rvs(size=dim_x, random_state=rng)
    sigma_68_scaled = sigma_68_prior.rvs(random_state=rng)
    nu = nu_prior.rvs(random_state=rng)
    return alpha_scaled, beta_scaled, sigma_68_scaled, nu

