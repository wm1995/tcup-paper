import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import scipy.stats as sps
from scipy.integrate import simpson
from tqdm import tqdm

from tcup_paper.plot import style

nus = np.logspace(-0.5, 2.5, 100)
num_datapoints = [10, 30, 100, 300]

rng = np.random.default_rng(0)

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
        xgrid = np.linspace(-10, 10, 10000) / np.sqrt(n)
        post = np.ones((2, nus.shape[0], xgrid.shape[0]))
        for idy, nu in enumerate(tqdm(nus)):
            # Generate dataset
            x = sps.norm.rvs(size=(n,), random_state=rng)

            z = x[:, np.newaxis] - xgrid[np.newaxis, :]

            prior = np.ones(xgrid.shape) # flat prior
            prior = sps.norm.pdf(xgrid, scale=3)

            stdev = np.zeros(2)
            for idz, dist in enumerate([sps.t(nu), sps.norm]):
                ll = np.sum(dist.logpdf(z), axis=0)
                post[idz, idy] = np.exp(ll) * prior
                post[idz, idy] /= (xgrid[1] - xgrid[0]) * post[idz, idy].sum()
                mean = np.average(xgrid, weights=post[idz, idy])
                stdev[idz] = np.sqrt(np.average((xgrid - mean) ** 2, weights=post[idz, idy]))

            # Check we've covered the necessary range
            assert np.isclose(simpson(post[idz, idy, :], x=xgrid), 1).all()

            # Calculate standard deviation ratio
            posterior_std_ratio[idx, idy] = stdev[0] / stdev[1]

    plt.figure(figsize=(10 / 3, 8 / 3))
    plt.semilogx(nus, posterior_std_ratio[2].T, "+")
    plt.xlim(0.25, 400)
    plt.ylim(0, 1.5)
    plt.axhline([1.0], color="k", linestyle="--")
    plt.xlabel(r"Shape parameter, $\nu$")
    plt.ylabel(r"Posterior standard deviation ratio")
    plt.tight_layout()
    plt.savefig(f"plots/posterior_sd.pdf")