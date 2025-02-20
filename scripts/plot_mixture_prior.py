import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from scipy.integrate import quad
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from tcup_paper.plot import style

seed = 20250211


def schechter(L, alpha, L0, phi):
    return phi * (L / L0) ** alpha * np.exp(-L / L0)


def double_power(L, L0, alpha, beta, phi):
    L_scaled = L / L0
    return phi / (L_scaled**alpha + L_scaled**beta)


def pdf_mixture_components(x, gmm):
    for w, mean, cov in zip(gmm.weights_, gmm.means_, gmm.covariances_):
        yield w * sps.multivariate_normal.pdf(x, mean=mean, cov=cov)


def pdf_mixture(x, gmm):
    return np.sum(
        np.fromiter(
            pdf_mixture_components(x, gmm),
            dtype=np.dtype((float, x.shape[0])),
        ),
        axis=0,
    )


plots = [
    {
        "type": "schechter",
        "name": "Schechter function",
        "lf": schechter,
        "lf_args": {
            "alpha": -1.25,
            "phi": 1e-3,
            "L0": 3e9,
        },
        "L_range": (1e5, 3e12),
        "plot_params": {
            "xlim": (1e6, 3e10),
            "ylim": (1e-16, 3e-7),
        },
    },
    {
        "type": "double_power",
        "name": "Double power law function",
        "lf": double_power,
        "lf_args": {
            "alpha": 0.5,
            "beta": 3,
            "phi": 1,
            "L0": 1e9,
        },
        "L_range": (1e5, 3e12),
        "plot_params": {
            "xlim": (1e6, 4e10),
            "ylim": (1e-14, 6e-8),
        },
    },
]

if __name__ == "__main__":
    style.apply_matplotlib_style()

    for params in plots:
        plt.figure(figsize=(10 / 3, 3))
        # Calculate normalisation over range
        norm = quad(
            lambda x: params["lf"](x, **params["lf_args"]),  # noqa: B023
            *params["L_range"],
        )[0]

        # Define normalised pdf and cdf
        def pdf(L):
            return params["lf"](L, **params["lf_args"]) / norm  # noqa: B023

        def cdf(L):
            return quad(
                lambda x: pdf(x),
                params["L_range"][0],  # noqa: B023
                L,
            )[0]

        # Invert CDF approximately using interpolation
        L = np.logspace(
            np.log10(params["L_range"][0]),
            np.log10(params["L_range"][1]),
            100,
        )
        cdf_vals = np.array([cdf(L_i) for L_i in L])
        inverse_cdf_approx = interp1d(cdf_vals, np.log10(L))

        # Draw samples from the PDF using our approximate CDF
        rng = np.random.default_rng(seed)
        logL_samples = inverse_cdf_approx(
            sps.uniform.rvs(size=1000000, random_state=rng)
        )

        # Fit mixture model
        gmm = GaussianMixture(n_components=10, random_state=seed)
        gmm.fit(logL_samples[:, np.newaxis])

        # Plot LF and its mixture model approximation
        # NB We pick up a factor of L log(10) because we fit our mixture model
        # to the log of the luminosity
        plt.loglog(L, pdf(L), label=params["name"])
        plt.loglog(
            L,
            pdf_mixture(np.log10(L), gmm) / L / np.log(10),
            "--",
            label="Mixture model approx.",
        )
        for p_comp in pdf_mixture_components(np.log10(L), gmm):
            plt.loglog(L, p_comp / L / np.log(10), "k:")

        plt.xlim(*params["plot_params"]["xlim"])
        plt.ylim(*params["plot_params"]["ylim"])
        plt.legend()
        plt.xlabel(r"Luminosity, $L$")
        plt.ylabel(r"Luminosity function, $\Phi(L)$")
        plt.tight_layout()

        plt.savefig(f"plots/pdf_mixture_{params['type']}.pdf")
        plt.close()
