import argparse
import pathlib
import re
import warnings

import arviz as az
import xarray as xr
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    var_names = [
        "alpha",
        "beta",
        "sigma_68",
    ]
    if args.model in ["tcup", "tobs"]:
        var_names += ["nu"]

    results_path = pathlib.Path("results/fixed/") / args.model / args.dataset
    mcmc_regex = re.compile(r".*/\d{1,3}\.nc")
    mcmc_files = filter(
        lambda x: mcmc_regex.match(str(x)), results_path.glob("*.nc")
    )

    results = {
        "prior": None,
        "posterior": None,
    }

    for mcmc_file in tqdm(mcmc_files):
        try:
            mcmc = az.from_netcdf(mcmc_file)
        except FileNotFoundError:
            warnings.warn("File not found")
            continue

        for group in results:
            if results[group] is None:
                results[group] = az.extract(
                    mcmc, group=group, var_names=var_names, num_samples=1000
                )
            else:
                samples = az.extract(
                    mcmc, group=group, var_names=var_names, num_samples=1000
                )
                results[group] = xr.concat(
                    [results[group], samples], dim="sample"
                )

    for group, samples in results.items():
        samples = samples.reset_index(["sample", "chain", "draw"], drop=True)
        samples.to_netcdf(
            results_path / "results.nc",
            group=group,
            engine="h5netcdf",
            mode="a",
        )
