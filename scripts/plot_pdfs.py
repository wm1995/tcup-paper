from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.special as jspec
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax.distributions as tfp_stats
import tensorflow_probability.substrates.jax.math as tfp_math


@jax.jit
def peak_height(nu):
    log_t = 0.5 * jnp.log(2)
    log_t -= 0.5 * jnp.log(nu)
    log_t += jspec.gammaln((nu + 1) / 2)
    log_t -= jspec.gammaln(nu / 2)
    t = jnp.where(nu == 0.0, 0.0, jnp.exp(log_t))
    t = jnp.where(jnp.isinf(nu), 1.0, t)
    return t


@jax.jit
def outlier_frac(nu, outlier_sigma=3):
    normal_outlier_frac = 1 - jspec.erf(outlier_sigma / jnp.sqrt(2))
    omega = tfp_math.betainc(0.5 * nu, 0.5, nu / (nu + outlier_sigma**2))
    omega = jnp.where(nu == 0, 0.0, omega)
    omega = jnp.where(jnp.isinf(nu), normal_outlier_frac, omega)
    return omega


def pdf_inv_nu(nu, coord):
    grad_x = jnp.vectorize(jax.grad(coord))
    # P(nu) = P(theta) |dtheta / d_nu|
    # 1/nu = theta ~ U(0, 1)
    # I've already taken the absolute value below
    dtheta = 1 / nu**2
    if coord == peak_height:
        # The following is a weird hack for peak height only
        # In the limit where nu is large, we can end up with numerical errors
        # This leads to the true probability being underestimated
        # Therefore, let's clip at the limiting value
        P_nu = jnp.where(
            nu >= 1,
            jnp.clip(dtheta / jnp.abs(grad_x(nu)), a_min=4.0),
            0.0,
        )
    else:
        P_nu = jnp.where(nu >= 1, dtheta, 0.0) / jnp.abs(grad_x(nu))

    return P_nu


def pdf_invgamma(nu, coord, alpha, beta):
    grad_x = jnp.vectorize(jax.grad(coord))
    return tfp_stats.InverseGamma(alpha, beta).prob(nu) / jnp.abs(grad_x(nu))


def pdf_F18(nu, coord):
    a = 1.2
    nu_0 = 0.55
    grad_x = jnp.vectorize(jax.grad(coord))

    P_nu = ((nu / nu_0) ** (1.0 / 2.0 / a) + (nu / nu_0) ** (2.0 / a)) ** -a
    log_norm = (
        jspec.gammaln(a / 3) + jspec.gammaln(2 * a / 3) - jspec.gammaln(a)
    )
    norm = 2 / 3 * a * nu_0 * jnp.exp(log_norm)
    return P_nu / norm / jnp.abs(grad_x(nu))


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    # Set matplotlib style
    preamble = r"""
    \usepackage{unicode-math}
    \setmainfont{XITS-Regular.otf}
    \setmathfont{XITSMath-Regular.otf}
    """
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["pgf.preamble"] = preamble
    mpl.rcParams["pgf.rcfonts"] = False
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"

    coords = [
        {
            "name": "nu",
            "symbol": r"\nu",
            "func": lambda x: x,
            "interp": False,
            "range": jnp.logspace(-3, 3, 500),
            "xscale": "log",
            "yscale": "log",
            "xlim": (1e-3, 1e3),
            "ylim": (1e-6, 1e2),
        },
        {
            "name": "peak_height",
            "symbol": r"t",
            "func": peak_height,
            "interp": True,
            "range": jnp.linspace(0, 1, 1000),
            "xscale": "linear",
            "yscale": "linear",
            "xlim": (0, 1),
        },
        {
            "name": "outlier_frac",
            "symbol": r"\omega",
            "func": outlier_frac,
            "interp": True,
            "range": jnp.linspace(0.002700, 1, 1000),
            "xscale": "linear",
            "yscale": "log",
            "xlim": (0, 1),
            "ylim": (1e-3, 1e2),
        },
    ]

    priors = [
        (partial(pdf_invgamma, alpha=3, beta=10), r"This work"),
        (pdf_F18, "Feeney et al. 2018"),
        (partial(pdf_invgamma, alpha=2, beta=10), r"Ju\'arez \& Steel (2010)"),
        (partial(pdf_invgamma, alpha=1, beta=10), r"Ding (2014)"),
        (pdf_inv_nu, r"Gelman et al. (2013)"),  # p. 443
    ]

    for coord in coords:
        if coord["interp"]:
            nu_interp = jnp.logspace(-7, 7, 100000)
            x_interp = coord["func"](nu_interp)
            ind = jnp.argsort(x_interp)
            nu = jnp.interp(coord["range"], x_interp[ind], nu_interp[ind])
        else:
            nu = coord["range"]

        grad_x = jnp.vectorize(jax.grad(coord["func"]))

        for pdf, label in priors:
            x = coord["func"](nu)
            # prob = pdf(nu) / jnp.abs(grad_x(nu))
            prob = pdf(nu, coord["func"])
            cdf = jnp.cumsum(prob)
            cdf /= cdf[-1]
            sf = 1 - cdf
            plt.figure("pdf")
            plt.plot(x, prob, "-", label=label)
            plt.figure("cdf")
            plt.plot(x, cdf, label=label)
            plt.figure("sf")
            plt.plot(x, sf, label=label)

        for plot in ["pdf", "cdf", "sf"]:
            plt.figure(plot)
            plt.legend()
            plt.xlabel(rf"${coord['symbol']}$")
            if plot == "pdf":
                plt.ylabel(rf"$P({coord['symbol']})$")
            elif plot == "cdf":
                plt.ylabel(rf"$F({coord['symbol']})$")
            elif plot == "sf":
                plt.ylabel(rf"$1 - F({coord['symbol']})$")
                if coord["name"] == "outlier_frac":
                    plt.yscale("log")
                    plt.ylim(1e-3, 1)
            plt.xscale(coord["xscale"])
            plt.xlim(coord.get("xlim"))
            if plot == "pdf":
                plt.yscale(coord["yscale"])
                plt.ylim(coord.get("ylim"))
            plt.savefig(f"plots/{plot}_{coord['name']}.pdf", backend="pgf")
            plt.close()
