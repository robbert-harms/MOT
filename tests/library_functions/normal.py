import unittest

from scipy.special._ufuncs import ndtri
from scipy.stats import norm
import numpy as np
from numpy.testing import assert_allclose
from mot.library_functions import normal_pdf, normal_cdf, normal_ppf
from mot.library_functions.continuous_distributions.normal import _ndtri, normal_logpdf
from mot.lib.utils import cartesian


class test_NormalDistribution(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        xs = np.arange(-10, 10, 0.5)
        ys = np.arange(0.01, 0.99, 0.1)
        means = np.arange(-10, 10, 0.5)
        stds = np.arange(0.01, 10, 0.5)

        self.distribution_test_params = cartesian([xs, means, stds]).astype(np.float64)
        self.quantile_test_params = cartesian([ys, means, stds]).astype(np.float64)

    def test_pdf(self):
        python_results = np.zeros(self.distribution_test_params.shape[0])
        for ind in range(self.distribution_test_params.shape[0]):
            x = self.distribution_test_params[ind, 0]
            mean = self.distribution_test_params[ind, 1]
            std = self.distribution_test_params[ind, 2]

            python_results[ind] = norm.pdf(x, loc=mean, scale=std)

        opencl_results = normal_pdf().evaluate({
            'x': self.distribution_test_params[:, 0],
            'mean': self.distribution_test_params[:, 1],
            'std': self.distribution_test_params[..., 2]}, self.distribution_test_params.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-5, rtol=1e-5)

    def test_logpdf(self):
        python_results = np.zeros(self.distribution_test_params.shape[0])
        for ind in range(self.distribution_test_params.shape[0]):
            x = self.distribution_test_params[ind, 0]
            mean = self.distribution_test_params[ind, 1]
            std = self.distribution_test_params[ind, 2]

            python_results[ind] = norm.logpdf(x, loc=mean, scale=std)

        opencl_results = normal_logpdf().evaluate({
            'x': self.distribution_test_params[:, 0],
            'mean': self.distribution_test_params[:, 1],
            'std': self.distribution_test_params[..., 2]}, self.distribution_test_params.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-5, rtol=1e-5)

    def test_cdf(self):
        python_results = np.zeros(self.distribution_test_params.shape[0])
        for ind in range(self.distribution_test_params.shape[0]):
            x = self.distribution_test_params[ind, 0]
            mean = self.distribution_test_params[ind, 1]
            std = self.distribution_test_params[ind, 2]

            python_results[ind] = norm.cdf(x, loc=mean, scale=std)

        opencl_results = normal_cdf().evaluate({
            'x': self.distribution_test_params[:, 0],
            'mean': self.distribution_test_params[:, 1],
            'std': self.distribution_test_params[..., 2]}, self.distribution_test_params.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-5, rtol=1e-5)

    def test_quantile(self):
        python_results = np.zeros(self.quantile_test_params.shape[0])
        for ind in range(self.quantile_test_params.shape[0]):
            y = self.quantile_test_params[ind, 0]
            mean = self.quantile_test_params[ind, 1]
            std = self.quantile_test_params[ind, 2]

            python_results[ind] = norm.ppf(y, loc=mean, scale=std)

        opencl_results = normal_ppf().evaluate({
            'y': self.quantile_test_params[:, 0],
            'mean': self.quantile_test_params[:, 1],
            'std': self.quantile_test_params[..., 2]}, self.quantile_test_params.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-5, rtol=1e-5)

    def test_ndtri(self):
        test_params = np.arange(0.01, 0.99, 0.01)

        python_results = ndtri(test_params.astype(np.float64))
        cl_results = _ndtri().evaluate({'y': test_params}, test_params.shape[0])

        assert_allclose(np.nan_to_num(python_results), np.nan_to_num(cl_results), atol=1e-5, rtol=1e-5)
