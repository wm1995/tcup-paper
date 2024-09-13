import matplotlib as mpl


def apply_matplotlib_style():
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
