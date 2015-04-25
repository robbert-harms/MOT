#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pppe
----------------------------------

Tests for `pppe` module.
"""

import unittest

import numpy as np

from pppe.cl_routines.mapping.residual_calculator import ResidualCalculator
from pppe.cl_routines.optimizing.nmsimplex import NMSimplex
from pppe.cl_routines.optimizing.levenberg_marquardt import LevenbergMarquardt
from pppe.cl_routines.optimizing.powell import Powell
from pppe.cl_routines.optimizing.serial_optimizers import SerialBasinHopping
from pppe.cl_routines.optimizing.serial_optimizers import SerialLM
from pppe.cl_routines.optimizing.serial_optimizers import SerialNMSimplex
from pppe.cl_routines.optimizing.serial_optimizers import SerialPowell
from pppe.cl_routines.smoothing.gaussian import GaussianSmoother
from pppe.cl_routines.smoothing.mean import MeanSmoother
from pppe.cl_routines.smoothing.median import MedianSmoother
from pppe.models.examples import Rosenbrock, MatlabLSQNonlinExample


class TestRosenbrock(unittest.TestCase):

    def setUp(self):
        self.model = Rosenbrock(5)
        self.optimizers = (NMSimplex(), Powell(), SerialBasinHopping(), SerialNMSimplex(), SerialPowell())

    def test_model(self):
        for optimizer in self.optimizers:
            v = optimizer.minimize(self.model)
            for p in self.model.get_optimized_param_names():
                self.assertAlmostEqual(v[p], 1, places=4)


class TestLSQNonLinExample(unittest.TestCase):

    def setUp(self):
        self.model = MatlabLSQNonlinExample()
        self.optimizers = (SerialLM(), LevenbergMarquardt())
        self.residual_calc = ResidualCalculator()

    def test_model(self):
        for optimizer in self.optimizers:
            v = optimizer.minimize(self.model)
            res = self.residual_calc.calculate(self.model, v)
            s = 0
            for i in range(res.shape[1]):
                s += res[0, i]**2
            self.assertAlmostEqual(s, 124.3622, places=4)


class TestSmoothing(unittest.TestCase):

    def setUp(self):
        self.d1 = np.array([1, 2, 4, 2, 1], dtype=np.float64)
        self.d2 = np.eye(4)

    def test_median(self):
        smoother = MedianSmoother(2)
        s1 = smoother.smooth(self.d1)
        np.testing.assert_almost_equal(s1, np.array([2, 2, 2, 2, 2]))

        s2 = smoother.smooth(self.d2)
        np.testing.assert_almost_equal(s2, np.zeros((4, 4)))

    def test_mean(self):
        smoother = MeanSmoother(2)
        s1 = smoother.smooth(self.d1)
        np.testing.assert_almost_equal(s1, np.array([2 + 1/3.0, 2.25, 2, 2.25, 2 + 1/3.0]))

        s2 = smoother.smooth(self.d2)
        expected = np.ones((4, 4)) * 0.25
        expected[0, 0] = 1/3.0
        expected[0, 3] = 2/9.0
        expected[3, 0] = 2/9.0
        expected[3, 3] = 1/3.0
        np.testing.assert_almost_equal(s2, expected)

    def test_gaussian(self):
        smoother = GaussianSmoother(2, sigma=1.0)
        s1 = smoother.smooth(self.d1, mask=np.array([1, 1, 1, 1, 0]))
        s2 = smoother.smooth(self.d2)

        np.testing.assert_almost_equal(s1, [1.1089774, 2.135224, 2.6417738, 1.8910226, 0])

        expected = np.array([[0.22470613, 0.20994687, 0.10351076, 0.02661242],
                             [0.20994687, 0.28434043, 0.22325308, 0.10351076],
                             [0.10351076, 0.22325308, 0.28434043, 0.20994687],
                             [0.02661242, 0.10351076, 0.20994687, 0.22470613]])
        np.testing.assert_almost_equal(s2, expected)

if __name__ == '__main__':
    unittest.main()
