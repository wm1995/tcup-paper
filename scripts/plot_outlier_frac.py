import jax
import jax.numpy as jnp
import jax.scipy.special as jspec
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax.math as tfp_math


@jax.jit
def outlier_frac(nu, outlier_sigma=3):
    normal_outlier_frac = 1 - jspec.erf(outlier_sigma / jnp.sqrt(2))
    omega = tfp_math.betainc(0.5 * nu, 0.5, nu / (nu + outlier_sigma**2))
    omega = jnp.where(nu == 0, 0.0, omega)
    omega = jnp.where(jnp.isinf(nu), normal_outlier_frac, omega)
    return omega


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    # Set matplotlib style
    preamble = r"""
    \usepackage{newtxtext, newtxmath}
    """
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["backend"] = "pgf"
    mpl.rcParams["pgf.preamble"] = preamble
    mpl.rcParams["pgf.rcfonts"] = False
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"

    nu = jnp.logspace(-3, 3, 200)
    omega = outlier_frac(nu)

    plt.figure(figsize=(10 / 3, 2))
    plt.hlines([1 - jspec.erf(3 / jnp.sqrt(2))], 1e-3, 1e3, "grey", "dashed")
    plt.loglog(nu, omega)
    plt.xlim((1e-3, 1e3))
    plt.xlabel(r"Shape parameter $\nu$")
    plt.ylabel(r"Outlier fraction $\omega(\nu)$")
    plt.tight_layout()
    plt.savefig("plots/outlier_frac.pdf")
