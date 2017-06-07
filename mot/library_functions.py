import os
from pkg_resources import resource_filename

__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CLLibrary(object):

    def get_cl_code(self):
        """Get the function code for this library and for all its dependencies.

        Returns:
            str: The CL code for inclusion in a kernel.
        """
        raise NotImplementedError()


class SimpleCLLibrary(CLLibrary):

    def __init__(self, name, cl_code, dependencies=None):
        """Python wrapper for library CL code.

        Args:
            name (str): the name of this library, used to create the inclusion guards
            cl_code (str): the CL code for this library
            dependencies (list or tuple of CLLibrary): The list of CL libraries this function depends on
        """
        super(SimpleCLLibrary, self).__init__()
        self._name = name
        self._cl_code = cl_code
        self._dependencies = dependencies or {}

    def get_cl_code(self):
        return '''
            {dependencies}
            #ifndef {inclusion_guard_name}
            #define {inclusion_guard_name}
            {code}
            #endif // {inclusion_guard_name}
        '''.format(dependencies=self._get_cl_dependency_code(),
                   inclusion_guard_name='LIBRARY_FUNCTION_{}_CL'.format(self._name),
                   code=self._cl_code)

    def _get_cl_dependency_code(self):
        """Get the CL code for all the CL code for all the dependencies.

        Returns:
            str: The CL code with the actual code.
        """
        code = ''
        for d in self._dependencies:
            code += d.get_cl_code() + "\n"
        return code


class SimpleCLLibraryFromFile(SimpleCLLibrary):

    def __init__(self, name, cl_code_file, var_replace_dict=None, dependencies=None):
        """Create a CL function for a library function.

        These functions are not meant to be optimized, but can be used a helper functions in models.

        Args:
            name (str): The name of the CL function
            cl_code_file (str): The location of the code file
            var_replace_dict (dict): In the cl_code file these replacements will be made
                (using the % format function of Python)
            dependencies (list or tuple of CLLibrary): The list of cl libraries this function depends on
        """
        code = open(os.path.abspath(cl_code_file), 'r').read()

        if var_replace_dict is not None:
            code = code % var_replace_dict

        super(SimpleCLLibraryFromFile, self).__init__(name, code, dependencies)


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
