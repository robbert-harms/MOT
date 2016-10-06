#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mot
----------------------------------

Tests for `mot` module.
"""

import unittest

import numpy as np

import mot
from mot import configuration
from mot.cl_routines.mapping.residual_calculator import ResidualCalculator
from mot.cl_routines.optimizing.nmsimplex import NMSimplex
from mot.cl_routines.optimizing.levenberg_marquardt import LevenbergMarquardt
from mot.cl_routines.optimizing.powell import Powell
from mot.cl_routines.filters.gaussian import GaussianFilter
from mot.cl_routines.filters.mean import MeanFilter
from mot.cl_routines.filters.median import MedianFilter
from mot.model_building.models_examples import Rosenbrock, MatlabLSQNonlinExample


class CLRoutineTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(CLRoutineTestCase, self).__init__(*args, **kwargs)
        mot.configuration._config['compile_flags']['general'].update({
            '-cl-single-precision-constant': False
        })


class TestRosenbrock(CLRoutineTestCase):

    def setUp(self):
        self.model = Rosenbrock(5)
        self.optimizers = (NMSimplex(), Powell())

    def test_model(self):
        for optimizer in self.optimizers:
            v = optimizer.minimize(self.model)
            for p in self.model.get_optimized_param_names():
                self.assertAlmostEqual(float(v[p]), 1.0, places=3)


class TestLSQNonLinExample(CLRoutineTestCase):

    def setUp(self):
        self.model = MatlabLSQNonlinExample()
        self.optimizers = (LevenbergMarquardt(),)
        self.residual_calc = ResidualCalculator()

    def test_model(self):
        for optimizer in self.optimizers:
            v = optimizer.minimize(self.model)
            res = self.residual_calc.calculate(self.model, v)
            s = 0
            for i in range(res.shape[1]):
                s += res[0, i]**2
            self.assertAlmostEqual(s, 124.3622, places=4)


class TestFilters(CLRoutineTestCase):

    def setUp(self):
        self.d1 = np.array([1, 2, 4, 2, 1], dtype=np.float64)
        self.d2 = np.eye(4)

    def test_median(self):
        filter = MedianFilter(2)
        s1 = filter.filter(self.d1)
        np.testing.assert_almost_equal(s1, np.array([2, 2, 2, 2, 2]))

        s2 = filter.filter(self.d2)
        np.testing.assert_almost_equal(s2, np.zeros((4, 4)))

    def test_mean(self):
        filter = MeanFilter(2)
        s1 = filter.filter(self.d1)
        np.testing.assert_almost_equal(s1, np.array([2 + 1/3.0, 2.25, 2, 2.25, 2 + 1/3.0]))

        s2 = filter.filter(self.d2)
        expected = np.ones((4, 4)) * 0.25
        expected[0, 0] = 1/3.0
        expected[0, 3] = 2/9.0
        expected[3, 0] = 2/9.0
        expected[3, 3] = 1/3.0
        np.testing.assert_almost_equal(s2, expected)

    def test_gaussian(self):
        filter = GaussianFilter(2, sigma=1.0)
        s1 = filter.filter(self.d1, mask=np.array([1, 1, 1, 1, 0]))
        s2 = filter.filter(self.d2)

        np.testing.assert_almost_equal(s1, [1.1089774, 2.135224, 2.6417738, 1.8910226, 0])

        expected = np.array([[0.22470613, 0.20994687, 0.10351076, 0.02661242],
                             [0.20994687, 0.28434043, 0.22325308, 0.10351076],
                             [0.10351076, 0.22325308, 0.28434043, 0.20994687],
                             [0.02661242, 0.10351076, 0.20994687, 0.22470613]])
        np.testing.assert_almost_equal(s2, expected)


if __name__ == '__main__':
    unittest.main()
