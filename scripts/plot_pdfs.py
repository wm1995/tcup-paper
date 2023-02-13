from functools import partial
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt

from tcup.priors import (
    pdf_F18,
    pdf_F18reparam,
    pdf_cauchy,
    pdf_nu2,
    pdf_peak_height,
    pdf_inv_nu,
    pdf_invgamma,
    pdf_invgamma2,
)
from tcup.utils import peak_height, outlier_frac

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
        },
        {
            "name": "outlier_frac",
            "symbol": r"\omega",
            "func": outlier_frac,
            "interp": True,
            "range": jnp.linspace(0.002700, 1, 1000),
            "xscale": "linear",
            "yscale": "log",
            "ylim": (1e-3, 1e2),
        },
    ]

    priors = [
        (pdf_F18, "Feeney+18"),
        (pdf_F18reparam, r"$t_{approx} \sim U(0, 1)$"),
        (pdf_peak_height, r"$t \sim U(0, 1)$"),
        (pdf_cauchy, r"$t_{approx} \sim U(t(\nu = 1), 1)$"),
        (partial(pdf_peak_height, nu_min=1), r"$t \sim U(t(\nu = 1), 1)$"),
        (pdf_nu2, r"$t_{approx} \sim U(t(\nu = 2), 1)$"),
        (partial(pdf_peak_height, nu_min=2), r"$t \sim U(t(\nu = 2), 1)$"),
        (pdf_invgamma, r"$\nu \sim Inv-\Gamma(3, 10)$"),
        (pdf_invgamma2, r"$\nu \sim Inv-\Gamma(2, 6)$"),
        (pdf_inv_nu, r"$\nu^{-1} \sim U(0, 1)$"),
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
            plt.xscale(coord["xscale"])
            if plot == "pdf":
                plt.yscale(coord["yscale"])
                plt.ylim(coord.get("ylim"))
            plt.savefig(f"plots/{plot}_{coord['name']}.pdf", backend="pgf")
            plt.close()
