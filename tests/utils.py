import unittest

import numpy as np
import pyopencl as cl

from mot.lib.utils import device_type_from_string, device_supports_double, get_float_type_def, is_scalar, \
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
