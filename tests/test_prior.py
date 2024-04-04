import pytest

import jax
import jax.numpy as jnp
from numpyro.distributions import Normal
from numpyro.infer import Predictive
import scipy.stats as sps
from tcup.numpyro import model_builder
from tcup_paper.model import prior

THRESHOLD = 1e-3 # Set the p-value threshold


@pytest.fixture
def tcup_samples():
    rng_key = jax.random.PRNGKey(0)
    tcup_model = model_builder(Normal())
    tcup_sampler = Predictive(tcup_model, num_samples=100000)
    return tcup_sampler(
        rng_key,
        x_scaled = jnp.array([[0]]),
        y_scaled = jnp.array([[0]]),
        cov_x_scaled = jnp.array([[[1]]]),
        dy_scaled = jnp.array([[1]]),
    )

@pytest.mark.parametrize(
    "param,prior",
    [
        ("alpha_scaled", prior.alpha_prior),
        ("beta_scaled", prior.beta_prior),
        ("sigma_68_scaled", prior.sigma_prior),
        ("nu", prior.nu_prior),
    ])
def test_prior_dist(tcup_samples, param, prior):
    assert sps.kstest(tcup_samples[param], prior.cdf).pvalue > THRESHOLD