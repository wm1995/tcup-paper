import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
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

    # Reproduce Kelly:2007 Figure 10a
    plt.errorbar(
        data["logLbol"],
        data["Gamma"],
        data[["E_Gamma", "e_Gamma"]].values.T,
        data["e_logLbol"],
        fmt="r+",
    )
    plt.xlim(-3, 1)
    plt.ylim(-1, 4)
    plt.savefig("plots/Kelly_10a.pdf")
    plt.close()

    # Reproduce Kelly:2007 Figure 10b
    plt.plot(
        data["logLbol"],
        data["Gamma"],
        "r+",
    )
    plt.xlim(-2.5, 0.5)
    plt.ylim(0, 4)
    plt.savefig("plots/Kelly_10b.pdf")
    plt.close()
