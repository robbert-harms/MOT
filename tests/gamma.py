import unittest
from scipy.stats import gamma
import numpy as np
from numpy.testing import assert_allclose

from mot.library_functions import GammaCDF
from mot.utils import cartesian


class test_GammaFunctions(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(test_GammaFunctions, self).__init__(*args, **kwargs)

    def test_gamma_cdf(self):
        test_params = self._gamma_cdf_params().astype(dtype=np.float32)

        python_results = self._calculate_python(test_params)
        cl_results = self._calculate_cl(test_params)

        assert_allclose(np.nan_to_num(python_results), np.nan_to_num(cl_results), atol=1e-5, rtol=1e-5)

    def _gamma_cdf_params(self):
        """Get some test params to test the Gamma CDF
        """
        shapes = np.arange(0.1, 5, 0.2)
        scales = np.arange(0.1, 5, 0.2)
        xs = np.arange(0.1, 10, 0.2)
        arr = cartesian([shapes, scales, xs])
        return arr

    def _calculate_cl(self, test_params):
        test_params = test_params.astype(np.float64)
        return GammaCDF().evaluate([test_params[:, 0], test_params[:, 1], test_params[..., 2]])

    def _calculate_python(self, input_params):
        results = np.zeros(input_params.shape[0])

        for ind in range(input_params.shape[0]):
            shape = input_params[ind, 0]
            scale = input_params[ind, 1]
            x = input_params[ind, 2]

            results[ind] = gamma.cdf(x, shape, scale=scale)
        return results
