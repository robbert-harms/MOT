import unittest
from scipy.stats import gamma
from scipy.special import gammainc, gammaincc
import numpy as np
from numpy.testing import assert_allclose
from mot.library_functions import gamma_pdf
from mot.library_functions.continuous_distributions.gamma import igamc, igam, gamma_cdf, gamma_ppf, gamma_logpdf
from mot.lib.utils import cartesian


class test_GammaDistribution(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_pdf(self):
        shapes = np.arange(0.01, 20, 0.5)
        scales = np.arange(0.01, 10, 0.5)
        xs = np.arange(0, 10, 0.5)

        test_params = cartesian([xs, shapes, scales]).astype(np.float64)

        python_results = np.zeros(test_params.shape[0])
        for ind in range(test_params.shape[0]):
            x = test_params[ind, 0]
            shape = test_params[ind, 1]
            scale = test_params[ind, 2]

            python_results[ind] = gamma.pdf(x, shape, scale=scale)

        opencl_results = gamma_pdf().evaluate({
            'x': test_params[:, 0],
            'shape': test_params[:, 1],
            'scale': test_params[..., 2]}, test_params.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-7, rtol=1e-7)

    def test_logpdf(self):
        shapes = np.arange(0.1, 20, 0.5)
        scales = np.arange(0.1, 10, 0.5)
        xs = np.arange(0, 10, 0.5)

        test_params = cartesian([xs, shapes, scales]).astype(np.float64)

        python_results = np.zeros(test_params.shape[0])
        for ind in range(test_params.shape[0]):
            x = test_params[ind, 0]
            shape = test_params[ind, 1]
            scale = test_params[ind, 2]

            python_results[ind] = gamma.logpdf(x, shape, scale=scale)

        opencl_results = gamma_logpdf().evaluate({
            'x': test_params[:, 0],
            'shape': test_params[:, 1],
            'scale': test_params[..., 2]}, test_params.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-7, rtol=1e-7)

    def test_igamc(self):
        a_list = np.arange(0.01, 10, 0.05)
        x_list = np.arange(0, 10, 0.1)
        test_params = cartesian([a_list, x_list]).astype(np.float64)

        python_results = np.zeros(test_params.shape[0])
        for ind in range(test_params.shape[0]):
            python_results[ind] = gammaincc(test_params[ind, 0], test_params[ind, 1])

        opencl_results = igamc().evaluate({'a': test_params[:, 0], 'x': test_params[:, 1]}, test_params.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-7, rtol=1e-7)

    def test_igam(self):
        a_list = np.arange(0.01, 10, 0.05)
        x_list = np.arange(0, 10, 0.1)
        test_params = cartesian([a_list, x_list]).astype(np.float64)

        python_results = np.zeros(test_params.shape[0])
        for ind in range(test_params.shape[0]):
            python_results[ind] = gammainc(test_params[ind, 0], test_params[ind, 1])

        opencl_results = igam().evaluate({'a': test_params[:, 0], 'x': test_params[:, 1]}, test_params.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-7, rtol=1e-7)

    def test_cdf(self):
        shapes = np.arange(0.01, 20, 0.5)
        scales = np.arange(0.01, 10, 0.5)
        xs = np.arange(0.01, 10, 0.5)

        test_params = cartesian([xs, shapes, scales]).astype(np.float64)

        python_results = np.zeros(test_params.shape[0])
        for ind in range(test_params.shape[0]):
            x = test_params[ind, 0]
            shape = test_params[ind, 1]
            scale = test_params[ind, 2]

            python_results[ind] = gamma.cdf(x, shape, scale=scale)

        opencl_results = gamma_cdf().evaluate({
            'x': test_params[:, 0],
            'shape': test_params[:, 1],
            'scale': test_params[:, 2]}, test_params.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-7, rtol=1e-7)

    def test_quantile(self):
        ys = np.arange(0.01, 0.99, 0.1)
        shapes = np.arange(0.01, 20, 0.5)
        scales = np.arange(0.01, 10, 0.5)

        test_params = cartesian([ys, shapes, scales]).astype(np.float64)

        python_results = np.zeros(test_params.shape[0])
        for ind in range(test_params.shape[0]):
            y = test_params[ind, 0]
            shape = test_params[ind, 1]
            scale = test_params[ind, 2]

            python_results[ind] = gamma.ppf(y, shape, scale=scale)

        opencl_results = gamma_ppf().evaluate({
            'y': test_params[:, 0],
            'shape': test_params[:, 1],
            'scale': test_params[:, 2]}, test_params.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-7, rtol=1e-7)

