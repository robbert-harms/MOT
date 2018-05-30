#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mot
----------------------------------

Tests for `mot` module.
"""

import unittest
import numpy as np
from mot.cl_routines.optimizing.nmsimplex import NMSimplex
from mot.cl_routines.optimizing.levenberg_marquardt import LevenbergMarquardt
from mot.cl_routines.optimizing.powell import Powell
from mot.utils import NameFunctionTuple, convert_data_to_dtype

from mot.model_interfaces import OptimizeModelInterface


class CLRoutineTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(CLRoutineTestCase, self).__init__(*args, **kwargs)


class TestRosenbrock(CLRoutineTestCase):

    def setUp(self):
        super(TestRosenbrock, self).setUp()
        self.model = Rosenbrock(5)
        self.optimizers = (NMSimplex(), Powell(patience=5))

    def test_model(self):
        for optimizer in self.optimizers:
            output = optimizer.minimize(self.model, np.array([[3] * 5]))
            v = output.get_optimization_result()
            for ind in range(2):
                self.assertAlmostEqual(v[0, ind], 1, places=3)

        def get_initial_parameters(self):
            params = np.ones((1, self.n)) * 3
            return convert_data_to_dtype(params, 'double')


class TestLSQNonLinExample(CLRoutineTestCase):

    def setUp(self):
        super(TestLSQNonLinExample, self).setUp()
        self.model = MatlabLSQNonlinExample()
        self.optimizers = (LevenbergMarquardt(), Powell(patience_line_search=5), NMSimplex())

    def test_model(self):
        for optimizer in self.optimizers:
            output = optimizer.minimize(self.model, np.array([[0.3, 0.4]]))
            v = output.get_optimization_result()
            for ind in range(2):
                self.assertAlmostEqual(v[0, ind], 0.2578, places=3)


class Rosenbrock(OptimizeModelInterface):

    def __init__(self, n=5):
        """When optimized the parameters should all be equal to 1."""
        super(OptimizeModelInterface, self).__init__()
        self.n = n

    @property
    def name(self):
        return 'rosenbrock'

    def get_kernel_data(self):
        return {}

    def get_nmr_problems(self):
        return 1

    def get_nmr_observations(self):
        return self.n - 1

    def get_nmr_parameters(self):
        return self.n

    def get_pre_eval_parameter_modifier(self):
        func_name = '_modifyParameters'
        func = '''
            void ''' + func_name + '''(void* data, mot_float_type* x){
            }
        '''
        return NameFunctionTuple(func_name, func)

    def get_objective_per_observation_function(self):
        func_name = 'getObjectiveInstanceValue'
        func = '''
            mot_float_type ''' + func_name + '''(void* data, const mot_float_type* const x, uint observation_index){
                uint i = observation_index;
                return 100 * pown(x[i + 1] - pown(x[i], 2), 2) + pown(1 - x[i], 2);
            }
        '''
        return NameFunctionTuple(func_name, func)

    def get_lower_bounds(self):
        return [-np.inf] * self.n

    def get_upper_bounds(self):
        return [np.inf] * self.n

    def finalize_optimized_parameters(self, parameters):
        return parameters


class MatlabLSQNonlinExample(OptimizeModelInterface):

    def __init__(self):
        """When optimized the parameters should be close to [0.2578, 0.2578] or something with a similar 2 norm.

        See the matlab manual page at http://nl.mathworks.com/help/optim/ug/lsqnonlin.html for more information.
        (viewed at 2015-04-02).

        """
        super(OptimizeModelInterface, self).__init__()

    @property
    def name(self):
        return 'matlab_lsqnonlin_example'

    def get_kernel_data(self):
        return {}

    def get_nmr_problems(self):
        return 1

    def get_nmr_observations(self):
        return 10

    def get_nmr_parameters(self):
        return 2

    def get_pre_eval_parameter_modifier(self):
        func_name = '_modifyParameters'
        func = '''
            void ''' + func_name + '''(void* data, mot_float_type* x){
            }
        '''
        return NameFunctionTuple(func_name, func)

    def get_objective_per_observation_function(self):
        func_name = "getObjectiveInstanceValue"
        func = '''
            mot_float_type ''' + func_name + '''(void* data, const mot_float_type* const x, uint observation_index){
                uint k = observation_index;
                return pown(2 + 2 * (k+1) - exp((k+1) * x[0]) - exp((k+1) * x[1]), 2);
            }
        '''
        return NameFunctionTuple(func_name, func)

    def get_lower_bounds(self):
        return [0, 0]

    def get_upper_bounds(self):
        return [np.inf] * 2

    def finalize_optimized_parameters(self, parameters):
        return parameters


if __name__ == '__main__':
    unittest.main()
