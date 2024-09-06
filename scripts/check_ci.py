import argparse
import warnings

import arviz as az
import pandas as pd
import xarray as xr
from tqdm import trange

from tcup_paper.data.io import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("-n", "--num-repeats", default=400, type=int)
    args = parser.parse_args()

    var_names = [
        "alpha",
        "beta",
        "sigma_68",
    ]
    if args.model in ["tcup", "tobs"]:
        var_names += ["nu"]

    results = []

    for idx in trange(args.num_repeats):
        try:
            _, params = load_dataset(
                f"data/fixed/{args.dataset}/{idx+1}.json"
            )
            true_vals = {
                param_name: params.get(param_name)
                for param_name in var_names
            }
            if true_vals["sigma_68"] is None:
                # Need this check because some of the normal datasets use
                # the older sigma_int specification
                true_vals["sigma_68"] = params.get("sigma_int")
            mcmc = az.from_netcdf(
                f"results/fixed/{args.model}/{args.dataset}/{idx+1}.nc"
            )
        except FileNotFoundError:
            warnings.warn("File not found")
            continue

        summary = az.summary(mcmc, var_names=var_names, hdi_prob=0.95)
        
        ci_results = {
            "idx": idx + 1,
        }
        
        for param, true_value in true_vals.items():
            if true_value is None:
                continue
            elif param == "beta":
                if isinstance(true_value, int | float):
                    true_value = [true_value]
                for idx, beta_value in enumerate(true_value):
                    beta_param = f"beta[{idx}]"
                    ci_results[beta_param] = (
                        (summary.loc[beta_param]["hdi_2.5%"] < beta_value)
                        and
                        (summary.loc[beta_param]["hdi_97.5%"] > beta_value)
                    )
            else:
                ci_results[param] = (
                    (summary.loc[param]["hdi_2.5%"] < true_value)
                    and
                    (summary.loc[param]["hdi_97.5%"] > true_value)
                )

        results += [ci_results]
        
    results = pd.DataFrame.from_records(results, index="idx")
    print(f"For {args.dataset=}, {args.model=}:")
    print(results.mean())
