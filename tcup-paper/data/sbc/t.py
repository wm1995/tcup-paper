import scipy.stats as sps

from tcup.utils import sigma_68

from .dataset import Dataset

class StudentTDataset(Dataset):
    def __init__(self, seed, n_data, dim_x, nu=None):
        super().__init__(seed, n_data, dim_x)
        self.nu = nu

    def draw_params_from_prior(self):
        params = super().draw_params_from_prior()
        self.alpha, self.beta, self.sigma_68, nu = params
        if self.nu is None:
            self.nu = nu

    def calc_y_true(self):
        self.sigma = self.sigma_68 / sigma_68(self.nu)
        epsilon = sps.t(df=self.nu, scale=self.sigma).rvs(
            size=self.n_data, random_state=self.rng
        )
        super().calc_y_true(epsilon)

    def get_info_dict(self):
        info = super().get_info_dict()
        info["sigma"] = self.sigma.tolist()
        return info