import json
from pathlib import Path

import jinja2
import numpy as np
import pandas as pd


def load_dataset(filename):
    with open(filename, "r") as f:
        dataset = json.load(f)

    # Calculate number of datapoints
    N = len(dataset["data"]["y"])

    # Pull out loaded parameters
    params = dataset["params"]

    params["type"] = filename.name.split("_")[0]
    params["N"] = N

    # Calculate number of outliers
    params["outliers"] = np.sum(params.pop("outliers"))

    # Expand out nested dictionaries
    params |= params.pop("y_true_params")
    params |= {
        f"x_obs_{key}": value
        for key, value in params.pop("x_obs_params").items()
    }
    params |= {
        f"y_obs_{key}": value
        for key, value in params.pop("y_obs_params").items()
    }

    # Calculate x dimensions and expand out coefficients
    params["dim_x"] = len(params["beta"])
    params |= {
        f"beta_{idx}": value for idx, value in enumerate(params.pop("beta"))
    }
    return params


if __name__ == "__main__":
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates/"))

    datasets = []

    for dataset in Path("data/").glob("*.json"):
        params = load_dataset(dataset)
        datasets.append(params)

    datasets = pd.DataFrame.from_records(datasets)

    # Reorder columns
    cols = [
        "type",
        "N",
        "dim_x",
        "outliers",
        "x_obs_err",
        "y_obs_err",
        "sigma_int",
        "alpha",
        "beta_0",
        "beta_1",
        "beta_2",
    ]
    datasets = datasets[cols]
    datasets.sort_values(by=["dim_x", "outliers"], inplace=True)

    # Title case the "type" column
    datasets["type"] = datasets["type"].str.title()

    # Rename columns for better presentation
    cols = {
        "type": r"Type",
        "N": r"$N$",
        "dim_x": r"$\dim x$",
        "outliers": r"Outliers",
        "x_obs_err": r"$\sigma_{x}$",
        "y_obs_err": r"$\sigma_{y}$",
        "alpha": r"$\alpha$",
        "beta_0": r"$\beta_0$",
        "beta_1": r"$\beta_1$",
        "beta_2": r"$\beta_2$",
        "sigma_int": r"$\sigma_{\text{int}}$",
    }
    styler = (
        datasets[
            ~((datasets["type"] == "Linear") & (datasets["outliers"] == 0))
        ]
        .rename(columns=cols)
        .style
    )

    styler.format(na_rep="-", precision=2)
    styler.hide(level=0, axis=0)
    datasets_latex = styler.to_latex()

    template = env.get_template("datasets.tex")
    content = template.render(datasets=datasets, datasets_latex=datasets_latex)

    with open("datasets.tex", "w") as f:
        f.write(content)
