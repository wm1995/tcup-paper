import numpy as np
import scipy.stats as sps

from .dataset import Dataset


class OutlierDataset(Dataset):
    def __init__(self, seed, n_data, dim_x, outlier_sigma):
        super().__init__(seed, n_data, dim_x)
        self.outlier_sigma = outlier_sigma

    def draw_params_from_prior(self):
        params = super().draw_params_from_prior()
        self.alpha, self.beta, self.sigma_68, _ = params
        self.outlier_idx = np.random.choice(self.n_data, 1)

    def calc_y_true(self):
        epsilon = sps.norm(scale=self.sigma_68).rvs(
            size=self.n_data, random_state=self.rng
        )
        outlier_sgn = (-1) ** sps.bernoulli(p=0.5).rvs(random_state=self.rng)

        # We want the outlier to be outlier_sigma sigma from the plane
        # Need to calculate the normal vector to the plane
        #    y = alpha + beta . x
        # => 0 = beta . x - (y - alpha)
        # normal_vec = np.concatenate([self.beta, [-1]])
        # delta_y is then outlier_sigma * sigma * | normal_vec |
        dy = self.outlier_sigma * self.sigma_68  # * np.linalg.norm(normal_vec)
        epsilon[self.outlier_idx] = outlier_sgn * dy
        super().calc_y_true(epsilon)

    def get_info_dict(self):
        info = super().get_info_dict()

        outlier_mask = np.zeros((self.n_data,)).astype(bool)
        outlier_mask[self.outlier_idx] = True
        info["outlier_mask"] = outlier_mask.tolist()

        return info
