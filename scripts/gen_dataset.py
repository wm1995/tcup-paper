#!/usr/bin/env python
import argparse

import numpy as np
from tcup_paper.data.io import write_dataset

# Set up run parameters
SEED = 24601


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fit a model to data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--t-dist", action="store_true")
    group.add_argument("--outlier", action="store_true")
    group.add_argument("--gaussian-mix", action="store_true")
    group.add_argument("--laplace", action="store_true")
    group.add_argument("--lognormal", action="store_true")
    args = parser.parse_args()

    if args.t_dist:
        from tcup_paper.data.t import gen_dataset

        x, y, dx, dy, info = gen_dataset(SEED)
        data = {
            "x": x.tolist(),
            "y": y.tolist(),
            "dx": dx.tolist(),
            "dy": dy.tolist(),
        }
        write_dataset("t", data, info)

    if args.outlier:
        from tcup_paper.data.outlier import gen_dataset

        x, y, dx, dy, info = gen_dataset(SEED)
        data = {
            "x": x.tolist(),
            "y": y.tolist(),
            "dx": dx.tolist(),
            "dy": dy.tolist(),
        }
        write_dataset("outlier", data, info)

        # Exclude outlier and save
        idx = info["outlier_idx"]
        mask = np.ones(shape=y.shape, dtype=bool)
        mask[idx] = False
        no_outlier_data = {
            "x": x[mask].tolist(),
            "y": y[mask].tolist(),
            "dx": dx[mask].tolist(),
            "dy": dy[mask].tolist(),
        }
        write_dataset("normal", no_outlier_data, info)

    if args.gaussian_mix:
        from tcup_paper.data.gaussian_mix import gen_dataset

        x, y, dx, dy, info = gen_dataset(SEED)
        data = {
            "x": x.tolist(),
            "y": y.tolist(),
            "cov_x": dx.tolist(),
            "dy": dy.tolist(),
        }
        write_dataset("gaussian_mix", data, info)

    if args.laplace:
        from tcup_paper.data.laplace import gen_dataset

        x, y, dx, dy, info = gen_dataset(SEED)
        data = {
            "x": x.tolist(),
            "y": y.tolist(),
            "dx": dx.tolist(),
            "dy": dy.tolist(),
        }
        write_dataset("laplace", data, info)

    if args.lognormal:
        from tcup_paper.data.lognormal import gen_dataset

        x, y, dx, dy, info = gen_dataset(SEED)
        data = {
            "x": x.tolist(),
            "y": y.tolist(),
            "dx": dx.tolist(),
            "dy": dy.tolist(),
        }
        write_dataset("lognormal", data, info)
