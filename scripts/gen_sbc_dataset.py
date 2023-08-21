#!/usr/bin/env python
import argparse

import numpy as np
from tcup_paper.data.io import write_dataset
from tcup_paper.data.sbc import gen_dataset


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fit a model to data")
    parser.add_argument("--seed", type=int, default=0)
    distribution = parser.add_mutually_exclusive_group(required=True)
    distribution.add_argument("--t-dist", action="store_true")
    distribution.add_argument("--fixed-nu", type=float)
    distribution.add_argument("--outlier", action="store_true")
    distribution.add_argument("--gaussian-mix", action="store_true")
    distribution.add_argument("--laplace", action="store_true")
    distribution.add_argument("--lognormal", action="store_true")
    args = parser.parse_args()

    x_true_params = {
        "N": 12,
        "D": 2,
        "K": 2,
        "theta_mix": [0.75, 0.25],
        "mu_mix": [[0.5, -0.5], [-1.5, 1.5]],
        "sigma_mix": [[[0.25, -0.1], [-0.1, 0.25]], [[0.25, 0.1], [0.1, 0.25]]]
    }

    if args.t_dist:
        dist_params = {
            "name": "t",
        }
    elif args.fixed_nu:
        dist_params = {
            "name": "fixed",
            "nu": args.fixed_nu,
        }
    elif args.outlier:
        dist_params = {
            "name": "normal",
        }
    elif args.outlier:
        dist_params = {
            "name": "outlier",
            "outlier_idx": 10,
        }
    elif args.gaussian_mix:
        dist_params = {
            "name": "gaussian_mix",
            "outlier_prob": 0.1,
        }
    elif args.laplace:
        dist_params = {
            "name": "laplace",
        }
    elif args.lognormal:
        dist_params = {
            "name": "lognormal",
        }
    else:
        raise RuntimeError("Something unexpected has gone wrong")

    data, info = gen_dataset(args.seed, x_true_params, dist_params)

    write_dataset(f"{dist_params['name']}_{args.seed}", data, info)