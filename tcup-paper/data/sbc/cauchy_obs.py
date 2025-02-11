import numpy as np
import scipy.stats as sps

from .normal import NormalDataset


class CauchyObsDataset(NormalDataset):
    def observe_data(self):
        eps_x = np.array(
            [
                sps.multivariate_t.rvs(
                    loc=np.zeros((self.dim_x,)),
                    shape=cov,
                    df=1,
                    random_state=self.rng,
                )
                for cov in self.cov_x
            ]
        ).reshape(self.n_data, self.dim_x)
        self.x_obs = self.x_true + eps_x
        self.y_obs = sps.cauchy(loc=self.y_true, scale=self.dy).rvs(
            random_state=self.rng
        )
