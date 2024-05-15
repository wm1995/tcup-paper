import scipy.stats as sps
from scipy.optimize import root

from .dataset import Dataset

class LognormalDataset(Dataset):
    def draw_params_from_prior(self):
        def cred_int_68(s):
            dist = sps.lognorm(s=s)
            lower, upper = dist.ppf(sps.norm.cdf([-1, 1]))
            return upper - lower

        params = super().draw_params_from_prior()
        self.alpha, self.beta, self.sigma_68, _ = params
        soln = root(lambda x: cred_int_68(x) - self.sigma_68, self.sigma_68)
        if not soln.success:
            raise RuntimeError("Unable to calculate sigma_68")
        else:
            self.sigma = soln.x[0]

    def calc_y_true(self):
        epsilon = sps.lognorm(s=self.sigma).rvs(
            size=self.n_data, random_state=self.rng
        )
        super().calc_y_true(epsilon)

    def get_info_dict(self):
        info = super().get_info_dict()
        info["sigma_scaled"] = self.sigma.tolist()
        return info