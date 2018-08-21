import unittest
import numpy as np
from mot.library_functions.error_functions import erfi, dawson
import scipy.special
from numpy.testing import assert_allclose

__author__ = 'Robbert Harms'
__date__ = '2018-05-12'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class test_ErrorFunctions(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_erfi(self):
        x = np.linspace(-3, 3)

        python_results = scipy.special.erfi(x)
        opencl_results = erfi().evaluate({'x': x}, x.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-5, rtol=1e-5)

    def test_dawson(self):
        x = np.linspace(-15, 15, num=1000)

        python_results = scipy.special.dawsn(x)
        opencl_results = dawson().evaluate({'x': x}, x.shape[0])

        assert_allclose(opencl_results, python_results, atol=1e-5, rtol=1e-5)
