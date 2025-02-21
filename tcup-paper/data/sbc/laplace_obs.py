import numpy as np
import scipy.special as spec
import scipy.stats as sps

from .normal import NormalDataset

SIGMA_68_CORR = -np.log(1 - spec.erf(1 / np.sqrt(2)))


class LaplaceObsDataset(NormalDataset):
    def __init__(self, seed, n_data, dim_x):
        if dim_x != 1:
            raise ValueError("`dim_x` must be 1 for this dataset")
        super().__init__(seed, n_data, dim_x)

    def observe_data(self):
        if self.dim_x != 1:
            raise ValueError("`dim_x` must be 1 for this dataset")

        eps_x = np.array(
            [
                sps.laplace.rvs(
                    loc=0,
                    shape=np.sqrt(cov) / SIGMA_68_CORR,
                    random_state=self.rng,
                )
                for cov in self.cov_x
            ]
        ).reshape(self.n_data, self.dim_x)
        self.x_obs = self.x_true + eps_x
        self.y_obs = sps.laplace(
            loc=self.y_true, scale=self.dy / SIGMA_68_CORR
        ).rvs(random_state=self.rng)
