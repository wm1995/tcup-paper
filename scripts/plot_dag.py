import daft
import matplotlib as mpl
import matplotlib.pyplot as plt
from tcup_paper.plot.style import apply_matplotlib_style

if __name__ == "__main__":
    # Set matplotlib style
    apply_matplotlib_style()

    param_style = {
        "shape": "rectangle",
    }

    latent_style = {}

    determ_style = {
        "alternate": True,
    }

    data_style = {
        "observed": True,
        "shape": "rectangle",
    }

    # Instantiate the PGM.
    pgm = daft.PGM(dpi=200)

    # Hierarchical parameters.
    pgm.add_node("alpha", r"$\alpha$", 3, 3, **param_style)
    pgm.add_node("beta", r"$\beta^j$", 4.5, 3, **param_style)
    pgm.add_node("sigma_int", r"$\sigma_{\rm int}$", 1.5, 3, **param_style)
    pgm.add_node("t", r"$\nu$", 0.5, 2, **param_style)
    # pgm.add_node("xi", r"$\xi$", 6, 2.5, **param_style)

    # Latent variables.
    pgm.add_node("eps", r"$\epsilon_i$", 1.5, 2, **latent_style)
    pgm.add_node("x", r"$x_i^j$", 4.5, 2, **param_style)  # **latent_style)

    # Deterministic functions.
    pgm.add_node("y", r"$y_i$", 3, 1.25, **determ_style)

    # Data.
    pgm.add_node("y_obs", r"$\hat{y}_i$", 3, 0, **data_style)
    pgm.add_node("x_obs", r"$\hat{x}_i^j$", 4.5, 0, **data_style)
    pgm.add_node("cov_x", r"$\symbf{\Sigma}_i$", 4.5, -1, **data_style)
    pgm.add_node("sigma_y", r"$\sigma_{y,i}$", 3, -1, **data_style)

    # Add in the edges.
    pgm.add_edge("alpha", "y")
    pgm.add_edge("beta", "y")
    # pgm.add_edge("xi", "x")
    pgm.add_edge("x", "y")
    pgm.add_edge("sigma_int", "eps")
    pgm.add_edge("eps", "y")
    pgm.add_edge("y", "y_obs")
    pgm.add_edge("x", "x_obs")
    pgm.add_edge("t", "eps")
    pgm.add_edge("cov_x", "x_obs")
    pgm.add_edge("sigma_y", "y_obs")

    # And a plate.
    pgm.add_plate(
        [1, -1.3, 4.4, 3.8],
        label=r"Obs. $i = 1:N$",
        position="bottom right",
        shift=-0.35,
    )
    pgm.add_plate(
        [3.75, -0.4, 1.5, 4],
        label=r"Vars. $j = 1:K$",
        position="top left",
        shift=-0.1,
    )

    # Render and save.
    pgm.render()
    plt.savefig("plots/dag.pdf", backend="pgf")
