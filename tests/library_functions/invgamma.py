import unittest
from scipy.stats import invgamma
import numpy as np
from numpy.testing import assert_allclose
from mot.library_functions import invgamma_pdf, invgamma_logpdf, invgamma_cdf, invgamma_ppf
from mot.lib.utils import cartesian


class test_GammaDistribution(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_pdf(self):
        shapes = np.arange(0.01, 20, 0.5)
        scales = np.arange(0.01, 10, 0.5)
        xs = np.arange(0.001, 10, 0.5)

        test_params = cartesian([xs, shapes, scales]).astype(np.float64)

        python_results = np.zeros(test_params.shape[0])
        for ind in range(test_params.shape[0]):
            x = test_params[ind, 0]
            shape = test_params[ind, 1]
            scale = test_params[ind, 2]

            python_results[ind] = invgamma.pdf(x, shape, scale=scale)

        opencl_results = invgamma_pdf().evaluate({
            'x': test_params[:, 0],
            'shape': test_params[:, 1],
            'scale': test_params[..., 2]}, test_params.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-7, rtol=1e-7)

    def test_logpdf(self):
        shapes = np.arange(0.1, 20, 0.5)
        scales = np.arange(0.1, 10, 0.5)
        xs = np.arange(0.001, 10, 0.5)

        test_params = cartesian([xs, shapes, scales]).astype(np.float64)

        python_results = np.zeros(test_params.shape[0])
        for ind in range(test_params.shape[0]):
            x = test_params[ind, 0]
            shape = test_params[ind, 1]
            scale = test_params[ind, 2]

            python_results[ind] = invgamma.logpdf(x, shape, scale=scale)

        opencl_results = invgamma_logpdf().evaluate({
            'x': test_params[:, 0],
            'shape': test_params[:, 1],
            'scale': test_params[..., 2]}, test_params.shape[0])

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

            python_results[ind] = invgamma.cdf(x, shape, scale=scale)

        opencl_results = invgamma_cdf().evaluate({
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

            python_results[ind] = invgamma.ppf(y, shape, scale=scale)

        opencl_results = invgamma_ppf().evaluate({
            'y': test_params[:, 0],
            'shape': test_params[:, 1],
            'scale': test_params[:, 2]}, test_params.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-7, rtol=1e-7)
