import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps

if __name__ == "__main__":
    # Set matplotlib style
    preamble = r"""
    \usepackage{newtxtext, newtxmath}
    """
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["pgf.preamble"] = preamble
    mpl.rcParams["pgf.rcfonts"] = False
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"

    nu_list = [0.1, 1, 3, 10, 30, np.inf]
    x = np.linspace(-5, 5, 200)

    plt.figure(figsize=(10 / 3, 3))
    for nu in nu_list:
        y = sps.t(df=nu).pdf(x)
        display_nu = r"\infty" if nu == np.inf else nu
        plt.plot(x, y, label=rf"$\nu = {display_nu}$")

    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t_{\nu}(x)$")

    plt.tight_layout()
    plt.savefig("plots/t-dist.pdf", backend="pgf")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("plots/t-dist-log.pdf", backend="pgf")
