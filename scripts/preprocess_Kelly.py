import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The following are pulled by digging around in the Postscript file
ORIGIN = (3330, 3080)
AXES = (19690, 14260)

# These are from the plot itself
XLIMS = (-3, 1)
YLIMS = (-1, 4)

if __name__ == "__main__":
    with open("data/kelly/f10a.ps", "r") as f:
        filetext = f.read()

    matches = re.findall(
        r"(\d+) \d+ M 0 (\d+) R D \d+ (\d+) M (\d+) 0 R D\s\d+ \d+ M 24[12] 0 R D\s\d+ \d+ M 24[12] 0 R D\s\d+ \d+ M 0 24[12] R D\s\d+ \d+ M 0 24[12] R D",
        filetext,
    )

    df = (
        pd.DataFrame(matches, columns=["x", "dy", "y", "dx"])
        .astype(int)
        .drop_duplicates()
    )

    xscale = (XLIMS[1] - XLIMS[0]) / AXES[0]
    df["log_Lbol"] = (df["x"] - ORIGIN[0]) * xscale + XLIMS[0]
    df["dlog_Lbol"] = df["dx"] * xscale / 2

    yscale = (YLIMS[1] - YLIMS[0]) / AXES[1]
    df["Gamma_X"] = (df["y"] - ORIGIN[1]) * yscale + YLIMS[0]
    df["dGamma_X"] = df["dy"] * yscale / 2

    # Reproduce Kelly:2007 Figure 10a
    plt.errorbar(
        df["log_Lbol"],
        df["Gamma_X"],
        df["dGamma_X"],
        df["dlog_Lbol"],
        fmt="r+",
    )
    plt.xlim(-3, 1)
    plt.ylim(-1, 4)
    plt.savefig("plots/Kelly_10a.pdf")
    plt.close()

    # Reproduce Kelly:2007 Figure 10b
    plt.plot(
        df["log_Lbol"],
        df["Gamma_X"],
        "r+",
    )
    plt.xlim(-2.5, 0.5)
    plt.ylim(0, 4)
    plt.savefig("plots/Kelly_10b.pdf")
    plt.close()

    data = {
        "x": df["log_Lbol"].values[:, np.newaxis].tolist(),
        "dx": df["dlog_Lbol"].values[:, np.newaxis, np.newaxis].tolist(),
        "y": df["Gamma_X"].values.tolist(),
        "dy": df["dGamma_X"].values.tolist(),
    }

    with open("data/kelly.json", "w") as f:
        json.dump(
            {
                "data": data,
                "params": {
                    "outliers": np.zeros(df["y"].shape, dtype=bool).tolist()
                },
            },
            f,
        )
else:
    # The else statement is just a hacky temporary way of skipping this block
    # Load spectral data from 2007ApJ...657..116K Table 4
    spectral = pd.read_fwf(
        "data/2007ApJ...657..116K_table4.dat",
        names=[
            "source",
            "new",
            "z",
            "NH",
            "n_0",
            "E_n0",
            "e_n0",
            "Gamma",
            "E_Gamma",
            "e_Gamma",
            "logL2500",
            "logL2kev",
            "a-ox",
            "a-UV",
        ],
    )

    # Select the data we need
    spectral = spectral.filter(
        items=[
            "source",
            "z",
            "Gamma",
            "E_Gamma",
            "e_Gamma",
            "logL2500",
        ]
    )
    spectral = spectral[spectral["z"] < 0.83]  # Filter redshift
    spectral = spectral.dropna()  # Drop points with no measured spectral index

    # Load masses & Eddington ratios from 2008ApJS..176..355K Table 1
    masses = pd.read_fwf(
        "data/2008ApJS..176..355K_table1.dat",
        names=[
            "RAh",
            "RAm",
            "RAs",
            "DEd",
            "DEm",
            "DEs",
            "z",
            "logMBL",
            "e_logMBL",
            "logLX",
            "e_logLX",
            "logLUV",
            "e_logLUV",
        ],
    )

    # Synthesise source column from RA and Dec
    masses["source"] = (
        masses["RAh"].apply(lambda x: f"{x:02}")
        + masses["RAm"].apply(lambda x: f"{x:02}")
        + masses["DEd"].apply(lambda x: f"{x:+03}")
        + masses["DEm"].apply(lambda x: f"{x:02}")
    )

    # Select only columns we need
    masses = masses.filter(
        items=["source", "z", "logMBL", "logLUV", "e_logLUV"]
    )

    # Join tables to create dataset
    data = spectral.merge(masses, on="source")

    # Ensure redshifts match
    data = data[np.isclose(data["z_x"], data["z_y"], atol=1e-3)]

    # Calculate bolometric luminosity
    bc = 5.6
    d_bc = 3.1
    d_log_bc = 1 / np.log(10) * d_bc / bc
    data["logLbol"] = data["logLUV"] + np.log10(bc)
    data["e_logLbol"] = np.sqrt(data["e_logLUV"] ** 2 + d_log_bc**2)
