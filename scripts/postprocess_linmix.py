import argparse
import json

import arviz as az
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    with open(args.input) as f:
        raw_data = json.load(f)

        cleaned_data = {}
        for key, value in raw_data.items():
            value_array = np.expand_dims(np.array(value), axis=0)

            match key:
                case "mu0":
                    cleaned_data["mix_mean"] = value_array
                case "usqr":
                    cleaned_data["mix_var"] = value_array
                case "pi":
                    cleaned_data["mix_weights"] = value_array
                case "mu":
                    cleaned_data["mix_means"] = value_array
                case "tausqr":
                    cleaned_data["mix_vars"] = value_array
                case "wsqr":
                    cleaned_data["mix_var_scale"] = value_array
                case "ximean":
                    cleaned_data["ximean"] = value_array
                case "xisig":
                    cleaned_data["xisig"] = value_array
                case "sigsqr":
                    cleaned_data["sigma_68"] = np.sqrt(value_array)
                case "alpha":
                    cleaned_data["alpha"] = value_array
                case "beta":
                    cleaned_data["beta"] = np.expand_dims(value_array, axis=-1)
                case "corr":
                    cleaned_data["corr_coeff"] = value_array
                case _:
                    raise KeyError("Unexpected key in linmix data")

        mcmc = az.from_dict(cleaned_data)
        mcmc.to_netcdf(args.output)