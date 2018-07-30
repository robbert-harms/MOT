#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mot
----------------------------------

Tests for `mot` module.
"""

import unittest
import numpy as np

from mot.cl_function import SimpleCLFunction
from mot.cl_routines.optimizing.nmsimplex import NMSimplex
from mot.cl_routines.optimizing.levenberg_marquardt import LevenbergMarquardt
from mot.cl_routines.optimizing.powell import Powell

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

    def get_nmr_observations(self):
        return self.n - 1

    def get_objective_function(self):
        return SimpleCLFunction.from_string('''
            double rosenbrock_MLE_func(mot_data_struct* data, const mot_float_type* const x,
                                       global mot_float_type* g_objective_list, 
                                       mot_float_type* p_objective_list,
                                       local double* objective_value_tmp){

                double sum = 0;
                double eval;
                for(uint i = 0; i < ''' + str(self.get_nmr_observations()) + '''; i++){
                    eval = 100 * pown(x[i + 1] - pown(x[i], 2), 2) + pown(1 - x[i], 2);
                    sum += eval;
                    
                    if(g_objective_list){
                        g_objective_list[i] = eval;
                    }
                    if(p_objective_list){
                        p_objective_list[i] = eval;
                    }
                }
                return sum;
            }
        ''')

    def get_lower_bounds(self):
        return [-np.inf] * self.n

    def get_upper_bounds(self):
        return [np.inf] * self.n


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

    def get_nmr_observations(self):
        return 10

    def get_objective_function(self):
        return SimpleCLFunction.from_string('''
            double lsqnonlin_example_objective(mot_data_struct* data, const mot_float_type* const x,
                                               global mot_float_type* g_objective_list, 
                                               mot_float_type* p_objective_list,
                                               local double* objective_value_tmp){
                
                double sum = 0;
                double eval;
                for(uint i = 0; i < ''' + str(self.get_nmr_observations()) + '''; i++){
                    eval = pown(2 + 2 * (i+1) - exp((i+1) * x[0]) - exp((i+1) * x[1]), 2);
                    sum += eval;
                    
                    if(g_objective_list){
                        g_objective_list[i] = eval;
                    }
                    if(p_objective_list){
                        p_objective_list[i] = eval;
                    }
                }
                return sum;
            }
        ''')

    def get_lower_bounds(self):
        return [0, 0]

    def get_upper_bounds(self):
        return [np.inf] * 2


if __name__ == '__main__':
    unittest.main()
