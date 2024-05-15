#!/usr/bin/env python
import argparse

from tcup_paper.data.io import write_dataset
import tcup_paper.data.sbc as sbc_data


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fit a model to data")
    parser.add_argument("--seed", type=int, default=0)
    distribution = parser.add_mutually_exclusive_group(required=True)
    distribution.add_argument("--t-dist", action="store_true")
    distribution.add_argument("--t-obs", action="store_true")
    distribution.add_argument("--fixed-nu", type=float)
    distribution.add_argument("--normal", action="store_true")
    distribution.add_argument("--outlier", type=float)
    distribution.add_argument("--gaussian-mix", action="store_true")
    distribution.add_argument("--laplace", action="store_true")
    distribution.add_argument("--lognormal", action="store_true")
    args = parser.parse_args()

    dataset_params = {
        "n_data": 12 if args.outlier else 50,
        "dim_x": 1,
        "seed": args.seed,
    }

    x_true_params = {
        "n_data": dataset_params["n_data"],
        "dim_x": 1,
        "weights": [1],
        "means": [[0]],
        "vars": [[[1]]],
    }

    if args.t_dist:
        dist_params = {
            "name": "t",
        }
        dataset = sbc_data.StudentTDataset(**dataset_params)
    elif args.t_obs:
        dist_params = {
            "name": "tobs",
        }
        dataset = sbc_data.TObsDataset(**dataset_params)
    elif args.fixed_nu:
        dist_params = {
            "name": "fixed",
            "nu": args.fixed_nu,
        }
        dataset = sbc_data.StudentTDataset(nu=args.fixed_nu, **dataset_params)
    elif args.normal:
        dist_params = {
            "name": "normal",
        }
        dataset = sbc_data.NormalDataset(**dataset_params)
    elif args.outlier:
        dist_params = {
            "name": "outlier",
            "outlier_sigma": args.outlier,
        }
        dataset = sbc_data.OutlierDataset(outlier_sigma=args.outlier, **dataset_params)
    elif args.gaussian_mix:
        dist_params = {
            "name": "gaussian_mix",
            "outlier_prob": 0.1,
        }
        dataset = sbc_data.GaussianMixDataset(outlier_prob=0.1, **dataset_params)
    elif args.laplace:
        dist_params = {
            "name": "laplace",
        }
        dataset = sbc_data.LaplaceDataset(**dataset_params)
    elif args.lognormal:
        dist_params = {
            "name": "lognormal",
        }
        dataset = sbc_data.LognormalDataset(**dataset_params)
    else:
        raise RuntimeError("Something unexpected has gone wrong")

    dataset.generate(x_true_params)

    if dist_params["name"] == "outlier":
        dataset.write(f"data/sbc/outlier{int(dist_params['outlier_sigma'])}/{args.seed}.json")
    else:
        dataset.write(f"data/sbc/{dist_params['name']}/{args.seed}.json")
