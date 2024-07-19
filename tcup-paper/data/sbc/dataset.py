import copy
import json
from abc import ABC, abstractmethod

import numpy as np
import scipy.stats as sps

from ...model.prior import draw_params_from_prior
from .utils import draw_x_true


class Dataset(ABC):
    def __init__(self, seed, n_data, dim_x):
        self.rng = np.random.default_rng(seed)
        self.n_data = n_data
        self.dim_x = dim_x

    @abstractmethod
    def draw_params_from_prior(self):
        return draw_params_from_prior(self.rng, self.dim_x)

    def draw_x_true(self, **params):
        self.x_true = draw_x_true(self.rng, **params)

    @abstractmethod
    def calc_y_true(self, epsilon):
        self.y_true = self.alpha + np.dot(self.x_true, self.beta) + epsilon

    def draw_errors(self):
        if self.dim_x == 1:
            dx = 10 ** sps.norm(loc=-1, scale=0.1).rvs(
                size=self.n_data,
                random_state=self.rng,
            )
            self.cov_x = (dx**2).reshape(self.n_data, self.dim_x, self.dim_x)
        else:
            self.cov_x = sps.wishart.rvs(
                self.dim_x + 1,
                np.diag([0.1] * self.dim_x),
                size=self.n_data,
                random_state=self.rng,
            ).reshape(self.n_data, self.dim_x, self.dim_x)
        log_dy = sps.norm(loc=-0.7, scale=0.2).rvs(
            size=self.n_data,
            random_state=self.rng,
        )
        self.dy = np.abs(self.beta) * 10**log_dy

    def observe_data(self):
        eps_x = np.array(
            [
                sps.multivariate_normal.rvs(
                    [0] * self.dim_x, cov, random_state=self.rng
                )
                for cov in self.cov_x
            ]
        ).reshape(self.n_data, self.dim_x)
        self.x_obs = self.x_true + eps_x
        self.y_obs = sps.norm(loc=self.y_true, scale=self.dy).rvs(
            random_state=self.rng
        )

    def generate(self, x_true_params=None):
        if x_true_params is None:
            self.x_true_params = {
                "n_data": self.n_data,
                "dim_x": self.dim_x,
                "weights": [1],
                "means": np.zeros((1, self.dim_x)),
                "vars": np.diag(np.ones(self.dim_x)).reshape(
                    1, self.dim_x, self.dim_x
                ),
            }
        else:
            self.x_true_params = copy.deepcopy(x_true_params)
            try:
                if self.x_true_params["n_data"] != self.n_data:
                    raise ValueError("`n_data` must match dataset")
            except KeyError:
                self.x_true_params["n_data"] = self.n_data

            try:
                if self.x_true_params["dim_x"] != self.dim_x:
                    raise ValueError("`dim_x` must match dataset")
            except KeyError:
                self.x_true_params["dim_x"] = self.dim_x

        self.draw_params_from_prior()
        self.draw_x_true(**self.x_true_params)
        self.calc_y_true()
        self.draw_errors()
        self.observe_data()

    def get_data_dict(self):
        data = copy.deepcopy(self.x_true_params)
        data["y_scaled"] = self.y_obs.tolist()
        data["dy_scaled"] = self.dy.tolist()
        data["x_scaled"] = self.x_obs.tolist()
        data["cov_x_scaled"] = self.cov_x.tolist()
        return data

    def get_info_dict(self):
        info = {
            "true_x": self.x_true.tolist(),
            "true_y": self.y_true.tolist(),
            "alpha_scaled": self.alpha.tolist(),
            "beta_scaled": self.beta.tolist(),
            "sigma_68_scaled": self.sigma_68.tolist(),
        }

        # Include nu if defined
        if hasattr(self, "nu"):
            info["nu"] = self.nu

        return info

    def write(self, path):
        output = {
            "data": self.get_data_dict(),
            "info": self.get_info_dict(),
        }
        with open(path, "w") as f:
            json.dump(output, f)
