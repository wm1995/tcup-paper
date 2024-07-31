import numpy as np
import pytest
import scipy.stats as sps
import tcup_paper.data.sbc as sbc

THRESHOLD = 1e-3


@pytest.fixture(params=[1, 2, 3])
def dataset_params(request):
    return {"seed": 0, "n_data": 10000, "dim_x": request.param}


true_x_params = {
    "weights": [0.75, 0.25],
    "means": [[0.5, -0.5], [-1.5, 1.5]],
    "covs": [[[0.25, -0.1], [-0.1, 0.25]], [[0.25, 0.1], [0.1, 0.25]]],
}


@pytest.mark.parametrize("nu", [None, 0.5, 1, 2, 10])
def test_student_t_dataset(dataset_params, nu):
    dataset = sbc.StudentTDataset(nu=nu, **dataset_params)
    dataset.generate()

    mu = dataset.alpha + np.dot(dataset.x_true, dataset.beta)
    t = (dataset.y_true - mu) / dataset.sigma

    if nu is not None:
        assert dataset.nu == nu

    assert sps.kstest(t, sps.t(df=dataset.nu).cdf).pvalue > THRESHOLD


def test_normal_dataset(dataset_params):
    dataset = sbc.NormalDataset(**dataset_params)
    dataset.generate()

    mu = dataset.alpha + np.dot(dataset.x_true, dataset.beta)
    z = (dataset.y_true - mu) / dataset.sigma_68

    assert sps.kstest(z, sps.norm.cdf).pvalue > THRESHOLD


@pytest.mark.parametrize("outlier_sigma", [1, 3, 5, 10, 20])
def test_outlier_dataset(dataset_params, outlier_sigma):
    dataset = sbc.OutlierDataset(outlier_sigma=outlier_sigma, **dataset_params)
    dataset.generate()

    mu = dataset.alpha + np.dot(dataset.x_true, dataset.beta)
    z = (dataset.y_true - mu) / dataset.sigma_68

    info = dataset.get_info_dict()
    outlier_mask = np.array(info["outlier_mask"])
    outlier = dataset.y_true[dataset.outlier_idx]
    outlier_logpdf = sps.norm(
        loc=mu[dataset.outlier_idx], scale=dataset.sigma_68
    ).logpdf(outlier)

    assert sps.kstest(z[~outlier_mask], sps.norm.cdf).pvalue > THRESHOLD
    assert np.isclose(
        outlier_logpdf,
        -np.log(np.pi * 2) / 2
        - np.log(dataset.sigma_68)
        - outlier_sigma**2 / 2,
    )


def test_gaussian_mix_dataset(dataset_params):
    dataset = sbc.GaussianMixDataset(outlier_prob=0.1, **dataset_params)
    dataset.generate()

    mu = dataset.alpha + np.dot(dataset.x_true, dataset.beta)
    assert (
        sps.binomtest(
            k=(dataset.y_true - mu < dataset.sigma_68).sum(),
            n=dataset.n_data,
            p=sps.norm.cdf(1),
        ).pvalue
        > THRESHOLD
    )

    z = (dataset.y_true - mu) / dataset.sigma
    assert sps.kstest(z, dataset.cdf).pvalue > THRESHOLD
    assert (
        sps.kstest(z[dataset.outlier_mask], sps.norm(scale=10).cdf).pvalue
        > THRESHOLD
    )
    assert (
        sps.kstest(z[~dataset.outlier_mask], sps.norm.cdf).pvalue > THRESHOLD
    )


def test_laplace_dataset(dataset_params):
    dataset = sbc.LaplaceDataset(**dataset_params)
    dataset.generate()

    mu = dataset.alpha + np.dot(dataset.x_true, dataset.beta)
    z = (dataset.y_true - mu) / dataset.sigma

    assert (
        sps.binomtest(
            k=(dataset.y_true - mu < dataset.sigma_68).sum(),
            n=dataset.n_data,
            p=sps.norm.cdf(1),
        ).pvalue
        > THRESHOLD
    )

    assert sps.kstest(z, sps.laplace.cdf).pvalue > THRESHOLD


def test_lognormal_dataset(dataset_params):
    dataset = sbc.LognormalDataset(**dataset_params)
    dataset.generate()

    dist = sps.lognorm(s=dataset.sigma)

    mu = dataset.alpha + np.dot(dataset.x_true, dataset.beta)
    epsilon = dataset.y_true - mu

    lower, upper = dist.ppf(sps.norm.cdf([-1, 1]))

    lower_tail = epsilon < lower
    upper_tail = epsilon > upper
    assert (
        sps.binomtest(
            k=(lower_tail | upper_tail).sum(),
            n=dataset.n_data,
            p=sps.norm.sf(1) * 2,
        ).pvalue
        > THRESHOLD
    )

    assert sps.kstest(epsilon, dist.cdf).pvalue > THRESHOLD
