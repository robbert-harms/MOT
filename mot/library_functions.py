import os
from textwrap import indent, dedent

from pkg_resources import resource_filename

from mot.cl_function import CLFunction, SimpleCLFunction
from mot.cl_parameter import CLFunctionParameter

__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CLLibrary(CLFunction):
    pass


class SimpleCLLibrary(CLLibrary, SimpleCLFunction):

    def __init__(self, return_type, cl_function_name, parameter_list, cl_code, dependency_list=()):
        """Python wrapper for library CL code.

        Args:
            cl_code (str): the CL code for this library
        """
        super(SimpleCLLibrary, self).__init__(return_type, cl_function_name, parameter_list,
                                              dependency_list=dependency_list)
        self._cl_code = cl_code

    def get_cl_code(self):
        return dedent('''
            {dependencies}
            #ifndef {inclusion_guard_name}
            #define {inclusion_guard_name}
            {code}
            #endif // {inclusion_guard_name}
        '''.format(dependencies=indent(self._get_cl_dependency_code(), ' '*4*3),
                   inclusion_guard_name='LIBRARY_FUNCTION_{}_CL'.format(self.cl_function_name),
                   code=indent('\n' + self._cl_code.strip() + '\n', ' '*4*3)))


class SimpleCLLibraryFromFile(SimpleCLLibrary):

    def __init__(self, return_type, cl_function_name, parameter_list, cl_code_file,
                 var_replace_dict=None, dependency_list=()):
        """Create a CL function for a library function.

        These functions are not meant to be optimized, but can be used a helper functions in models.

        Args:
            cl_function_name (str): The name of the CL function
            cl_code_file (str): The location of the code file
            var_replace_dict (dict): In the cl_code file these replacements will be made
                (using the % format function of Python)
            dependency_list (list or tuple of CLLibrary): The list of cl libraries this function depends on
        """
        with open(os.path.abspath(cl_code_file), 'r') as f:
            code = f.read()

        if var_replace_dict is not None:
            code = code % var_replace_dict

        super(SimpleCLLibraryFromFile, self).__init__(return_type, cl_function_name, parameter_list, code,
                                                      dependency_list=dependency_list)


class FirstLegendreTerm(SimpleCLLibraryFromFile):

    def __init__(self):
        """A function for finding the first legendre term. (see the CL code for more details)"""
        super(FirstLegendreTerm, self).__init__(
            'double', 'firstLegendreTerm', [CLFunctionParameter('double', 'x'), CLFunctionParameter('int', 'n')],
            resource_filename('mot', 'data/opencl/firstLegendreTerm.cl'))


class Bessel(SimpleCLLibraryFromFile):

    def __init__(self):
        """Function library for the bessel functions."""
        super(Bessel, self).__init__(
            'double', 'bessel_i0', [CLFunctionParameter('double', 'x')],
            resource_filename('mot', 'data/opencl/bessel.cl'))


class GammaFunctions(SimpleCLLibraryFromFile):

    def __init__(self):
        """Function library for the bessel functions."""
        super(GammaFunctions, self).__init__(
            'double', 'gamma_p', [CLFunctionParameter('double', 'a'), CLFunctionParameter('double', 'x')],
            resource_filename('mot', 'data/opencl/gammaFunctions.cl'))


class LogCosh(SimpleCLLibraryFromFile):

    def __init__(self):
        """Estimate various trigonometric functions additional to the OpenCL offerings."""
        super(LogCosh, self).__init__(
            'double', 'log_cosh', [CLFunctionParameter('double', 'x')],
            resource_filename('mot', 'data/opencl/trigonometrics.cl'))


class Rand123(SimpleCLLibrary):

    def __init__(self):
        """Estimate various trigonometric functions additional to the OpenCL offerings."""
        super(Rand123, self).__init__('void', 'rand123', [], Rand123._get_random123_cl_code())

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
            'void', 'cerf', [],
            resource_filename('mot', 'data/opencl/cerf/im_w_of_x.cl'))


class CerfDawson(SimpleCLLibraryFromFile):

    def __init__(self):
        """Evaluate dawson integral."""
        super(CerfDawson, self).__init__(
            'void', 'dawson', [],
            resource_filename('mot', 'data/opencl/cerf/dawson.cl'),
            dependency_list=(CerfImWOfX(),))


class CerfErfi(SimpleCLLibraryFromFile):

    def __init__(self):
        """Calculate erfi."""
        super(CerfErfi, self).__init__(
            'void', 'erfi', [],
            resource_filename('mot', 'data/opencl/cerf/erfi.cl'),
            dependency_list=(CerfImWOfX(),))


class EuclidianNormFunction(SimpleCLLibraryFromFile):

    def __init__(self, memspace='private', memtype='mot_float_type'):
        """A CL functions for calculating the Euclidian distance between n values.

        Args:
            memspace (str): The memory space of the memtyped array (private, constant, global).
            memtype (str): the memory type to use, double, float, mot_float_type, ...
        """
        super(EuclidianNormFunction, self).__init__(
            memtype,
            self.__class__.__name__ + '_' + memspace + '_' + memtype,
            [],
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
            'int', 'lib_nmsimplex', [],
            resource_filename('mot', 'data/opencl/lib_nmsimplex.cl'),
            var_replace_dict=params)
