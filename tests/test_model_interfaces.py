import inspect
import unittest
from mot.model_interfaces import OptimizeModelInterface, SampleModelInterface

__author__ = 'Robbert Harms'
__date__ = "2017-03-28"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class test_OptimizeModelInterface(unittest.TestCase):

    def test_for_not_implemented_error(self):
        interface = OptimizeModelInterface()
        self.assertRaises(NotImplementedError, lambda: interface.name)
        self.assertRaises(NotImplementedError, lambda: interface.double_precision)

        functions = inspect.getmembers(OptimizeModelInterface, predicate=inspect.isfunction)
        for function in functions:
            sig = inspect.signature(function[1])
            extra_args = [None]*(len(sig.parameters) -1)
            self.assertRaises(NotImplementedError, function[1], interface, *extra_args)


class test_SampleModelInterface(unittest.TestCase):

    def test_for_not_implemented_error(self):
        interface = SampleModelInterface()

        functions = inspect.getmembers(SampleModelInterface, predicate=inspect.isfunction)
        for function in functions:
            sig = inspect.signature(function[1])
            extra_args = [None] * (len(sig.parameters) - 1)
            self.assertRaises(NotImplementedError, function[1], interface, *extra_args)
