#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mot
----------------------------------

Tests for `mot` module.
"""

import unittest
import numpy as np

from mot import minimize
from mot.lib.cl_function import SimpleCLFunction

from mot.lib.model_interfaces import OptimizeModelInterface


class CLRoutineTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(CLRoutineTestCase, self).__init__(*args, **kwargs)


class TestRosenbrock(CLRoutineTestCase):

    def setUp(self):
        super(TestRosenbrock, self).setUp()
        self.model = Rosenbrock(5)
        self.methods = {'Nelder-Mead': None, 'Powell': {'patience': 3}}

    def test_model(self):
        for method, options in self.methods.items():
            output = minimize(self.model, np.array([[3] * 5]), method=method, options=options)
            v = output['x']
            for ind in range(2):
                self.assertAlmostEqual(v[0, ind], 1, places=3, msg=method)


class TestLSQNonLinExample(CLRoutineTestCase):

    def setUp(self):
        super(TestLSQNonLinExample, self).setUp()
        self.model = MatlabLSQNonlinExample()
        self.methods = ('Levenberg-Marquardt', 'Powell', 'Nelder-Mead')

    def test_model(self):
        for method in self.methods:
            output = minimize(self.model, np.array([[0.3, 0.4]]), method=method)
            v = output['x']
            for ind in range(2):
                self.assertAlmostEqual(v[0, ind], 0.2578, places=3, msg=method)


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
            double rosenbrock_MLE_func(mot_data_struct* data, 
                                       local const mot_float_type* const x,
                                       local mot_float_type* objective_list,
                                       local double* objective_value_tmp){

                double sum = 0;
                double eval;
                for(uint i = 0; i < ''' + str(self.get_nmr_observations()) + '''; i++){
                    eval = 100 * pown(x[i + 1] - pown(x[i], 2), 2) + pown(1 - x[i], 2);
                    sum += eval;
                    
                    if(objective_list){
                        objective_list[i] = eval;
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
            double lsqnonlin_example_objective(mot_data_struct* data, 
                                               local const mot_float_type* const x,
                                               local mot_float_type* objective_list, 
                                               local double* objective_value_tmp){
                
                double sum = 0;
                double eval;
                for(uint i = 0; i < ''' + str(self.get_nmr_observations()) + '''; i++){
                    eval = pown(2 + 2 * (i+1) - exp((i+1) * x[0]) - exp((i+1) * x[1]), 2);
                    sum += eval;
                    
                    if(objective_list){
                        objective_list[i] = eval;
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
