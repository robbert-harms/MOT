import unittest
from textwrap import dedent

import numpy as np
import pyopencl as cl
from numpy.testing import assert_array_equal

from mot.utils import device_type_from_string, device_supports_double, results_to_dict, get_float_type_def, is_scalar, \
    all_elements_equal, get_single_value, topological_sort

__author__ = 'Robbert Harms'
__date__ = "2017-03-28"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class test_device_type_from_string(unittest.TestCase):

    def test_gpu(self):
        assert(device_type_from_string('GPU') == cl.device_type.GPU)

    def test_cpu(self):
        assert(device_type_from_string('CPU') == cl.device_type.CPU)

    def test_accelerator(self):
        assert(device_type_from_string('ACCELERATOR') == cl.device_type.ACCELERATOR)

    def test_custom(self):
        assert(device_type_from_string('CUSTOM') == cl.device_type.CUSTOM)

    def test_none(self):
        assert(device_type_from_string('') is None)


class test_device_supports_double(unittest.TestCase):

    def test_has_double(self):
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                has_double = device.get_info(cl.device_info.DOUBLE_FP_CONFIG) == 63
                assert(device_supports_double(device) == has_double)


class test_results_to_dict(unittest.TestCase):

    def test_mismatch(self):
        results = np.zeros((2, 3, 4))
        param_names = ['only_one_name_for_three_params']
        self.assertRaises(ValueError, results_to_dict, results, param_names)

    def test_2d_matrix(self):
        results = np.random.rand(2, 3)
        param_names = ['p1', 'p2', 'p3']
        results_dict = results_to_dict(results, param_names)

        assert(all(name in results_dict for name in param_names))

        for ind, name in enumerate(param_names):
            assert_array_equal(results_dict[name], results[:, ind])

    def test_3d_matrix(self):
        results = np.random.rand(2, 3, 4)
        param_names = ['p1', 'p2', 'p3']
        results_dict = results_to_dict(results, param_names)

        assert(all(name in results_dict for name in param_names))

        for ind, name in enumerate(param_names):
            assert_array_equal(results_dict[name], results[:, ind, :])


class test_get_float_type_def(unittest.TestCase):

    def test_float(self):
        known_good_value = '''
            #if __OPENCL_VERSION__ <= CL_VERSION_1_1
                #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            #endif

            #define mot_float_type float
            #define mot_float_type2 float2
            #define mot_float_type4 float4
            #define mot_float_type8 float8
            #define mot_float_type16 float16
            #define MOT_EPSILON FLT_EPSILON
            #define MOT_MIN FLT_MIN
            #define MOT_MAX FLT_MAX
            #define MOT_INT_CMP_TYPE int
        '''
        value = get_float_type_def(False)

        assert(dedent(value) == dedent(known_good_value))

    def test_double(self):
        known_good_value = '''
            #if __OPENCL_VERSION__ <= CL_VERSION_1_1
                #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            #endif

            #define mot_float_type double
            #define mot_float_type2 double2
            #define mot_float_type4 double4
            #define mot_float_type8 double8
            #define mot_float_type16 double16
            #define MOT_EPSILON DBL_EPSILON
            #define MOT_MIN DBL_MIN
            #define MOT_MAX DBL_MAX
            #define MOT_INT_CMP_TYPE long
        '''
        value = get_float_type_def(True)

        assert(dedent(value) == dedent(known_good_value))


class test_is_scalar(unittest.TestCase):

    def test_is_not_scalar(self):
        self.assertFalse(is_scalar(np.zeros((2, 2))))

    def test_is_scalar(self):
        self.assertTrue(is_scalar(np.zeros((1, ))[:, None]))
        self.assertTrue(is_scalar(np.zeros((1,))))
        self.assertTrue(is_scalar(-1))
        self.assertTrue(is_scalar(0))
        self.assertTrue(is_scalar(1))
        self.assertTrue(is_scalar(-1.0))
        self.assertTrue(is_scalar(0.0))
        self.assertTrue(is_scalar(1.0))


class test_all_elements_equal(unittest.TestCase):

    def test_scalar(self):
        self.assertTrue(all_elements_equal(np.zeros((1,))[:, None]))
        self.assertTrue(all_elements_equal(np.zeros((1,))))
        self.assertTrue(all_elements_equal(-1))
        self.assertTrue(all_elements_equal(0))
        self.assertTrue(all_elements_equal(1))
        self.assertTrue(all_elements_equal(-1.0))
        self.assertTrue(all_elements_equal(0.0))
        self.assertTrue(all_elements_equal(1.0))

    def test_false(self):
        self.assertFalse(all_elements_equal(np.random.rand(2, 2)))

    def test_matrix(self):
        self.assertTrue(all_elements_equal(np.zeros((2,))))
        self.assertTrue(all_elements_equal(np.zeros((2, 3))))
        self.assertTrue(all_elements_equal(np.zeros((2, 3, 4))))


class test_get_single_value(unittest.TestCase):

    def test_exception(self):
        self.assertRaises(ValueError, get_single_value, np.random.rand(2, 2))

    def test_true(self):
        self.assertTrue(get_single_value(np.ones((2, 2))[:, None]) == 1)
        self.assertTrue(get_single_value(np.zeros((1,))[:, None]) == 0)
        self.assertTrue(get_single_value(np.zeros((1,))) == 0)
        self.assertTrue(get_single_value(-1) == -1)
        self.assertTrue(get_single_value(0) == 0)
        self.assertTrue(get_single_value(1) == 1)
        self.assertTrue(get_single_value(-1.0) == -1.0)
        self.assertTrue(get_single_value(0.0) == 0.0)
        self.assertTrue(get_single_value(1.0) == 1.0)


class test_topological_sort(unittest.TestCase):

    def test_auto_dependency(self):
        circular = {'a': ('a',), 'b': ('a',)}
        self.assertRaises(ValueError, topological_sort, circular)

    def test_cyclic_dependency(self):
        circular = {'a': ('b',), 'b': ('a',)}
        self.assertRaises(ValueError, topological_sort, circular)

    def test_sorting(self):
        data = {'a': (), 'm': ('c',), 'e': ('m',), '!': ('a', 'e', 'c')}
        assert(list(topological_sort(data)) == ['a', 'c', 'm', 'e', '!'])

    def test_unsortables(self):
        class A():
            pass

        class B():
            pass

        a = A()
        b = B()
        data = {'a': (a, b), 'b': ('a',)}

        assert(len(list(topological_sort(data))) == 4)

    def test_empty_input(self):
        data = {}
        self.assertFalse(topological_sort(data))
