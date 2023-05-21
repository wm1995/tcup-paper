import matplotlib as mpl


def apply_matplotlib_style():
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
