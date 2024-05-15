import scipy.stats as sps

from .dataset import Dataset

class LaplaceDataset(Dataset):
    def draw_params_from_prior(self):
        params = super().draw_params_from_prior()
        self.alpha, self.beta, self.sigma_68, _ = params
        self.sigma = self.sigma_68 / sps.laplace.ppf(sps.norm.cdf(1))

    def calc_y_true(self):
        epsilon = sps.laplace(scale=self.sigma).rvs(
            size=self.n_data, random_state=self.rng
        )
        super().calc_y_true(epsilon)

    def get_info_dict(self):
        info = super().get_info_dict()
        info["sigma_scaled"] = self.sigma.tolist()
        return info