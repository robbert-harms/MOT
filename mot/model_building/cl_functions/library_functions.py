import os
from pkg_resources import resource_filename
from mot.model_building.cl_functions.base import SimpleCLLibraryFromFile, SimpleCLLibrary

__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class FirstLegendreTerm(SimpleCLLibraryFromFile):

    def __init__(self):
        """A function for finding the first legendre term. (see the CL code for more details)"""
        super(FirstLegendreTerm, self).__init__(
            self.__class__.__name__, resource_filename('mot', 'data/opencl/firstLegendreTerm.cl'))


class Bessel(SimpleCLLibraryFromFile):

    def __init__(self):
        """Function library for the bessel functions."""
        super(Bessel, self).__init__(self.__class__.__name__, resource_filename('mot', 'data/opencl/bessel.cl'))


class Trigonometrics(SimpleCLLibraryFromFile):

    def __init__(self):
        """Estimate various trigonometric functions additional to the OpenCL offerings."""
        super(Trigonometrics, self).__init__(
            self.__class__.__name__, resource_filename('mot', 'data/opencl/trigonometrics.cl'))


class Rand123(SimpleCLLibrary):

    def __init__(self):
        """Estimate various trigonometric functions additional to the OpenCL offerings."""
        super(Rand123, self).__init__(self.__class__.__name__, Rand123._get_random123_cl_code())

    @staticmethod
    def _get_random123_cl_code():
        """Get the source code needed for working with the Rand123 RNG.

        Returns:
            str: the CL code for the Rand123 RNG
        """
        generator = 'threefry'

        src = open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/openclfeatures.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/array.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/{}.h'.format(generator)), ),
                    'r').read()
        src += (open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/rand123.h'), ), 'r').read() % {
            'GENERATOR_NAME': (generator)
        })
        return src


class CerfImWOfX(SimpleCLLibraryFromFile):

    def __init__(self):
        """Calculate the cerf."""
        super(CerfImWOfX, self).__init__(
            self.__class__.__name__, resource_filename('mot', 'data/opencl/cerf/im_w_of_x.cl'))


class CerfDawson(SimpleCLLibraryFromFile):

    def __init__(self):
        """Evaluate dawson integral."""
        super(CerfDawson, self).__init__(
            self.__class__.__name__, resource_filename('mot', 'data/opencl/cerf/dawson.cl'),
            dependencies=(CerfImWOfX(),))


class CerfErfi(SimpleCLLibraryFromFile):

    def __init__(self):
        """Calculate erfi."""
        super(CerfErfi, self).__init__(
            self.__class__.__name__, resource_filename('mot', 'data/opencl/cerf/erfi.cl'),
            dependencies=(CerfImWOfX(),))


class EuclidianNormFunction(SimpleCLLibraryFromFile):

    def __init__(self, memspace='private', memtype='mot_float_type'):
        """A CL functions for calculating the Euclidian distance between n values.

        Args:
            memspace (str): The memory space of the memtyped array (private, constant, global).
            memtype (str): the memory type to use, double, float, mot_float_type, ...
        """
        super(EuclidianNormFunction, self).__init__(
            self.__class__.__name__ + '_' + memspace + '_' + memtype,
            resource_filename('mot', 'data/opencl/euclidian_norm.cl'),
            var_replace_dict={'MEMSPACE': memspace, 'MEMTYPE': memtype})


class LibNMSimplex(SimpleCLLibraryFromFile):

    def __init__(self, evaluate_fname='evaluate'):
        """A CL functions for calculating the Euclidean distance between n values.

        Args:
            evaluate_fname (str): the name of the evaluation function to use, default 'evaluate'
        """
        params = {
            'EVALUATE_FNAME': str(evaluate_fname)
        }

        super(LibNMSimplex, self).__init__(
            'lib_nmsimplex',
            resource_filename('mot', 'data/opencl/lib_nmsimplex.cl'),
            var_replace_dict=params)
