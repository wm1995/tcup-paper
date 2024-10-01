import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import scipy.stats as sps
from tqdm import tqdm

from tcup_paper.plot import style

nus = jnp.logspace(-0.5, 2.5, 100)
num_datapoints = [100] # [10, 30, 100, 300]

rng = np.random.default_rng(0)
rng_key = jax.random.PRNGKey(0)

def norm_model(data):
    mu = numpyro.sample("mu", dist.Normal(0.0, 3.0))
    sigma = 1
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=data)

def t_model(nu, data):
    mu = numpyro.sample("mu", dist.Normal(0.0, 3.0))
    sigma = 1
    numpyro.sample("obs", dist.StudentT(nu, mu, sigma), obs=data)

if __name__ == "__main__":
    style.apply_matplotlib_style()

    posterior_std_ratio = np.zeros((len(num_datapoints), len(nus)))
    for idx, n in enumerate(num_datapoints):
        print(f"For {n=}")
        for idy, nu in enumerate(tqdm(nus)):
            # Generate dataset
            x = sps.norm.rvs(size=(n,), random_state=rng)

            # Run normal model
            rng_key, rng_key_ = jax.random.split(rng_key)

            kernel = NUTS(norm_model)
            num_samples = 2000
            mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples, progress_bar=False)
            mcmc.run(
                rng_key_, data=x
            )
            samples_norm = mcmc.get_samples()

            # Run t model
            rng_key, rng_key_ = jax.random.split(rng_key)

            kernel = NUTS(t_model)
            num_samples = 2000
            mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples, progress_bar=False)
            mcmc.run(
                rng_key_, nu=nu, data=x
            )
            samples_t = mcmc.get_samples()

            # Calculate standard deviation ratio
            posterior_std_ratio[idx, idy] = (samples_t["mu"].std() / samples_norm["mu"].std())

    plt.figure(figsize=(10 / 3, 3))
    plt.semilogx(nus, posterior_std_ratio.T, "+")
    plt.xlim(0.25, 400)
    plt.ylim(0.5, 1.5)
    plt.axhline([1.0], color="k", linestyle="--")
    plt.xlabel(r"Shape parameter, $\nu$")
    plt.ylabel(r"Posterior standard deviation ratio")
    plt.tight_layout()
    plt.savefig(f"plots/posterior_sd.pdf")