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
from mot.cl_routines.optimizing.nmsimplex import NMSimplex
from mot.cl_routines.optimizing.levenberg_marquardt import LevenbergMarquardt
from mot.cl_routines.optimizing.powell import Powell
from mot.cl_routines.filters.gaussian import GaussianFilter
from mot.cl_routines.filters.mean import MeanFilter
from mot.cl_routines.filters.median import MedianFilter
from mot.utils import SimpleNamedCLFunction, convert_data_to_dtype

from mot.model_interfaces import OptimizeModelInterface


class CLRoutineTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(CLRoutineTestCase, self).__init__(*args, **kwargs)
        self._old_config_value = mot.configuration._config['compile_flags']['general']['-cl-single-precision-constant']

    def setUp(self):
        mot.configuration._config['compile_flags']['general'].update({
            '-cl-single-precision-constant': False
        })

    def tearDown(self):
        mot.configuration._config['compile_flags']['general'].update({
            '-cl-single-precision-constant': self._old_config_value
        })


class TestRosenbrock(CLRoutineTestCase):

    def setUp(self):
        super(TestRosenbrock, self).setUp()
        self.model = Rosenbrock(5)
        self.optimizers = (NMSimplex(), Powell(patience=10))

    def test_model(self):
        for optimizer in self.optimizers:
            v = optimizer.minimize(self.model).get_optimization_result()[0]
            for ind in range(self.model.get_nmr_inst_per_problem()):
                self.assertAlmostEqual(float(v[ind]), 1.0, places=3)


class TestLSQNonLinExample(CLRoutineTestCase):

    def setUp(self):
        super(TestLSQNonLinExample, self).setUp()
        self.model = MatlabLSQNonlinExample()
        self.optimizers = (LevenbergMarquardt(),)
        self.residual_calc = ResidualCalculator()

    def test_model(self):
        for optimizer in self.optimizers:
            v = optimizer.minimize(self.model).get_optimization_result()
            res = self.residual_calc.calculate(self.model, v)
            s = 0
            for i in range(res.shape[1]):
                s += res[0, i]**2
            self.assertAlmostEqual(s, 124.3622, places=4)


class TestFilters(CLRoutineTestCase):

    def setUp(self):
        super(TestFilters, self).setUp()
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



class Rosenbrock(OptimizeModelInterface):

    def __init__(self, n=5):
        """When optimized the parameters should all be equal to 1."""
        super(OptimizeModelInterface, self).__init__()
        self.n = n

    def double_precision(self):
        return True

    @property
    def name(self):
        return 'rosenbrock'

    def get_kernel_data(self):
        return []

    def get_nmr_problems(self):
        return 1

    def get_pre_eval_parameter_modifier(self):
        func_name = '_modifyParameters'
        func = '''
            void ''' + func_name + '''(void* data, mot_float_type* x){
            }
        '''
        return SimpleNamedCLFunction(func, func_name)

    def get_model_eval_function(self):
        fname = 'evaluateModel'
        func = '''
            double ''' + fname + '''(void* data, const double* const x, uint observation_index){
                double sum = 0;
                for(uint i = 0; i < ''' + str(self.n) + ''' - 1; i++){
                    sum += 100 * pown((x[i + 1] - pown(x[i], 2)), 2) + pown((x[i] - 1), 2);
                }
                return -sum;
            }
        '''
        return SimpleNamedCLFunction(func, fname)

    def _get_observation_return_function(self):
        fname = 'getObservation'
        func = '''
            double ''' + fname + '''(void* data, const uint observation_index){
                return 0;
            }
        '''
        return SimpleNamedCLFunction(func, fname)

    def get_objective_per_observation_function(self):
        eval_func = self.get_model_eval_function()
        obs_func = self._get_observation_return_function()

        func = eval_func.get_cl_code()
        func += obs_func.get_cl_code()

        func_name = "getObjectiveInstanceValue"
        func += '''
            mot_float_type ''' + func_name + '''(void* data, const mot_float_type* const x, uint observation_index){
                return ''' + obs_func.get_cl_function_name() + '''(data, observation_index) -
                            ''' + eval_func.get_cl_function_name() + '''(data, x, observation_index);
            }
        '''
        return SimpleNamedCLFunction(func, func_name)

    def get_initial_parameters(self):
        params = np.ones((1, self.n)) * 3
        return convert_data_to_dtype(params, 'double')

    def get_lower_bounds(self):
        return [-np.inf] * self.n

    def get_upper_bounds(self):
        return [np.inf] * self.n

    def get_free_param_names(self):
        return list(map(str, range(self.n)))

    def get_nmr_inst_per_problem(self):
        return 1

    def get_nmr_estimable_parameters(self):
        return self.n

    def finalize_optimized_parameters(self, parameters):
        return parameters

class MatlabLSQNonlinExample(OptimizeModelInterface):

    def __init__(self):
        """When optimized the parameters should be close to [0.2578, 0.2578] or something with a similar 2 norm.

        See the matlab manual page at http://nl.mathworks.com/help/optim/ug/lsqnonlin.html for more information.
        (viewed at 2015-04-02).

        """
        super(OptimizeModelInterface, self).__init__()

    def double_precision(self):
        return True

    @property
    def name(self):
        return 'matlab_lsqnonlin_example'

    def get_kernel_data(self):
        return []

    def get_nmr_problems(self):
        return 1

    def get_pre_eval_parameter_modifier(self):
        func_name = '_modifyParameters'
        func = '''
            void ''' + func_name + '''(void* data, mot_float_type* x){
            }
        '''
        return SimpleNamedCLFunction(func, func_name)

    def get_model_eval_function(self):
        fname = 'evaluateModel'
        func = '''
            double ''' + fname + '''(void* data, const double* const x, uint k){
                return -(2 + 2 * (k+1) - exp((k+1) * x[0]) - exp((k+1) * x[1]));
            }
        '''
        return SimpleNamedCLFunction(func, fname)

    def _get_observation_return_function(self):
        fname = 'getObservation'
        func = '''
            double ''' + fname + '''(void* data, uint observation_index){
                return 0;
            }
        '''
        return SimpleNamedCLFunction(func, fname)

    def get_objective_per_observation_function(self):
        eval_func = self.get_model_eval_function()
        obs_func = self._get_observation_return_function()

        func = eval_func.get_cl_code()
        func += obs_func.get_cl_code()

        func_name = "getObjectiveInstanceValue"

        func += '''
            mot_float_type ''' + func_name + '''(void* data, const mot_float_type* const x, uint observation_index){
                return ''' + obs_func.get_cl_function_name() + '''(data, observation_index) -
                            ''' + eval_func.get_cl_function_name() + '''(data, x, observation_index);
            }
        '''
        return SimpleNamedCLFunction(func, func_name)

    def get_initial_parameters(self):
        params = np.array([[0.3, 0.4]])
        return convert_data_to_dtype(params, 'double')

    def get_lower_bounds(self):
        return [0, 0]

    def get_upper_bounds(self):
        return [np.inf] * 2

    def get_free_param_names(self):
        return ['0', '1']

    def get_nmr_inst_per_problem(self):
        return 10

    def get_nmr_estimable_parameters(self):
        return 2

    def finalize_optimized_parameters(self, parameters):
        return parameters

if __name__ == '__main__':
    unittest.main()
