import scipy.stats as sps

from .dataset import Dataset

class NormalDataset(Dataset):
    def draw_params_from_prior(self):
        params = super().draw_params_from_prior()
        self.alpha, self.beta, self.sigma_68, _ = params

    def calc_y_true(self):
        epsilon = sps.norm(scale=self.sigma_68).rvs(
            size=self.n_data, random_state=self.rng
        )
        super().calc_y_true(epsilon)