import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import jax.scipy.stats as jsps
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tensorflow_probability.substrates.jax.math import erfcx
from tqdm import tqdm

from tcup_paper.plot import style

jax.config.update("jax_enable_x64", True)

@jax.jit
def fisher_pred(nu):
    f = jnp.sqrt(nu / 2)
    q = (nu + 1) * (1 - jnp.sqrt(jnp.pi) * f * erfcx(f))
    return 1 / jnp.sqrt(q)

if __name__ == "__main__":
    style.apply_matplotlib_style()
    rng_key = jax.random.key(0)
    dists = [jsps.t, jsps.norm]
    nus = jnp.logspace(-1, 2.5, 100)
    ratio = jnp.zeros(nus.shape)
    xgrid = jnp.linspace(-2, 2, 3000)[:, jnp.newaxis]
    num_datapoints = 100
    post = jnp.ones((xgrid.shape[0], 2))

    for idx, nu in enumerate(tqdm(nus)):
        rng_key, rng_key_ = jax.random.split(rng_key)
        x = jax.random.normal(rng_key_, shape=(1, num_datapoints))
        z = x - xgrid

        prior = jnp.ones((xgrid.shape[0]))

        ll = jnp.sum(jsps.t.logpdf(z, nu), axis=1)
        ll -= ll.max(axis=0)

        ll_norm = jnp.sum(jsps.norm.logpdf(z), axis=1)
        ll_norm -= ll_norm.max(axis=0)

        post = post.at[:, 0].set(jnp.exp(ll) * prior)
        post = post.at[:, 1].set(jnp.exp(ll_norm) * prior)
        post /= (xgrid[1] - xgrid[0]) * post.sum(axis=0)

        mean = jnp.sum(xgrid * post, axis=0) / jnp.sum(post, axis=0)
        stdev = jnp.sqrt(jnp.sum((xgrid - mean)**2 * post, axis=0)) / jnp.sum(post, axis=0)

        ratio = ratio.at[idx].set(stdev[0] / stdev[1])

    plt.figure(figsize=(10 / 3, 8 / 3))
    plt.semilogx(nus, ratio, "+")
    plt.semilogx(nus, fisher_pred(nus), "k--")
    plt.xlim(0.2, 300)
    plt.ylim(0, 1.5)
    plt.axhline([1.0], color="k", linestyle="--", alpha=0.5)
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.xlabel(r"Shape parameter, $\nu$")
    plt.ylabel(r"Posterior standard deviation ratio")
    plt.tight_layout()
    plt.savefig(f"plots/posterior_sd.pdf")