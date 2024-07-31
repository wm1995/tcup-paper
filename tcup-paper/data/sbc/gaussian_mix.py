import numpy as np
import scipy.stats as sps
from scipy.optimize import root

from .dataset import Dataset


class GaussianMixDataset(Dataset):
    def __init__(self, seed, n_data, dim_x, outlier_prob):
        super().__init__(seed, n_data, dim_x)
        self.outlier_prob = outlier_prob

    def pdf(self, x):
        core_pdf = sps.norm.pdf(x)
        outlier_pdf = sps.norm(scale=10).pdf(x)
        p_core = 1 - self.outlier_prob
        p_outlier = self.outlier_prob
        return p_core * core_pdf + p_outlier * outlier_pdf

    def cdf(self, x):
        core_cdf = sps.norm.cdf(x)
        outlier_cdf = sps.norm(scale=10).cdf(x)
        p_core = 1 - self.outlier_prob
        p_outlier = self.outlier_prob
        return p_core * core_cdf + p_outlier * outlier_cdf

    def ppf_68(self):
        soln = root(lambda x: self.cdf(x) - sps.norm.cdf(1), 1, jac=self.pdf)
        if soln.success:
            return soln.x[0]
        else:
            raise RuntimeError("Failed to solve for sigma_68")

    def draw_params_from_prior(self):
        params = super().draw_params_from_prior()
        self.alpha, self.beta, self.sigma_68, _ = params
        self.sigma = self.sigma_68 / self.ppf_68()

    def calc_y_true(self):
        self.outlier_mask = (
            sps.bernoulli(self.outlier_prob)
            .rvs(size=self.n_data, random_state=self.rng)
            .astype(bool)
        )

        int_scatter = sps.norm(scale=self.sigma).rvs(
            size=self.n_data, random_state=self.rng
        )
        epsilon = np.where(self.outlier_mask, 10 * int_scatter, int_scatter)
        super().calc_y_true(epsilon)

    def get_info_dict(self):
        info = super().get_info_dict()
        info["sigma_scaled"] = self.sigma.tolist()
        info["outlier_mask"] = self.outlier_mask.tolist()
        return info
