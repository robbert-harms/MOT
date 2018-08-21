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


class CLRoutineTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TestRosenbrock(CLRoutineTestCase):

    def setUp(self):
        super().setUp()
        self.n = 5
        self.methods = {'Nelder-Mead': None, 'Powell': {'patience': 3}}
        self._nmr_observations = self.n - 1
        self._objective_func = SimpleCLFunction.from_string('''
            double rosenbrock_MLE_func(local const mot_float_type* const x,
                                       void* data, 
                                       local mot_float_type* objective_list){

                double sum = 0;
                double eval;
                for(uint i = 0; i < ''' + str(self.n - 1) + '''; i++){
                    eval = 100 * pown(x[i + 1] - pown(x[i], 2), 2) + pown(1 - x[i], 2);
                    sum += eval;

                    if(objective_list){
                        objective_list[i] = eval;
                    }
                }
                return sum;
            }
        ''')

    def test_model(self):
        for method, options in self.methods.items():
            output = minimize(self._objective_func, np.array([[3] * 5]), method=method,
                              nmr_observations=self._nmr_observations, options=options)
            v = output['x']
            for ind in range(2):
                self.assertAlmostEqual(v[0, ind], 1, places=3, msg=method)


class TestLSQNonLinExample(CLRoutineTestCase):

    def setUp(self):
        super().setUp()
        self._nmr_observations = 10
        self._objective_func = SimpleCLFunction.from_string('''
            double lsqnonlin_example_objective(local const mot_float_type* const x,
                                               void* data, 
                                               local mot_float_type* objective_list){
                
                double sum = 0;
                double eval;        
                for(uint i = 0; i < ''' + str(self._nmr_observations) + '''; i++){
                    eval = pown(2 + 2 * (i+1) - exp((i+1) * x[0]) - exp((i+1) * x[1]), 2);
                    sum += eval;
                    
                    if(objective_list){
                        objective_list[i] = eval;
                    }
                }
                return sum;
            }
        ''')
        self.methods = ('Levenberg-Marquardt', 'Powell', 'Nelder-Mead')

    def test_model(self):
        for method in self.methods:
            output = minimize(self._objective_func, np.array([[0.3, 0.4]]), method=method,
                              nmr_observations=self._nmr_observations)
            v = output['x']
            for ind in range(2):
                self.assertAlmostEqual(v[0, ind], 0.2578, places=3, msg=method)


if __name__ == '__main__':
    unittest.main()
