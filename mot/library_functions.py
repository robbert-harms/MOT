import os
from textwrap import indent, dedent

from pkg_resources import resource_filename
from mot.cl_function import CLFunction, SimpleCLFunction

__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CLLibrary(CLFunction):
    pass


class SimpleCLLibrary(CLLibrary, SimpleCLFunction):
    pass


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
        self._code = code

    def get_cl_code(self):
        return dedent('''
            {dependencies}
            #ifndef {inclusion_guard_name}
            #define {inclusion_guard_name}
            {cl_extra}
            {code}
            #endif // {inclusion_guard_name}
        '''.format(dependencies=indent(self._get_cl_dependency_code(), ' ' * 4 * 3),
                   inclusion_guard_name='INCLUDE_GUARD_{}'.format(self.get_cl_function_name()),
                   cl_extra=self._cl_extra if self._cl_extra is not None else '',
                   code=indent('\n' + self._code.strip() + '\n', ' ' * 4 * 3)))


class FirstLegendreTerm(SimpleCLLibrary):

    def __init__(self):
        """Compute the first term of the legendre polynome for the given value x and the polynomial degree n.

        The Legendre polynomials, Pn(x), are orthogonal on the interval [-1,1] with weight function w(x) = 1
        for -1 <= x <= 1 and 0 elsewhere.  They are normalized so that Pn(1) = 1.  The inner products are:

        .. code-block:: c

            <Pn,Pm> = 0        if n != m,
            <Pn,Pn> = 2/(2n+1) if n >= 0.


        This routine calculates Pn(x) using the following recursion:

        .. code-block:: c

            (k+1) P[k+1](x) = (2k+1)x P[k](x) - k P[k-1](x), k = 1,...,n-1
            P[0](x) = 1, P[1](x) = x.


        The function arguments are:

        * x: The argument of the Legendre polynomial Pn.
        * n: The degree of the Legendre polynomial Pn.

        The return value is Pn(x) if n is a nonnegative integer.  If n is negative, 0 is returned.
        """
        super(FirstLegendreTerm, self).__init__(
            'double', 'firstLegendreTerm',
            [('double', 'x'), ('int', 'n')],
            '''
                if (n < 0){
                    return 0.0;
                }

                if(fabs(x) == 1.0){
                    if(x > 0.0 || n % 2 == 0){
                        return 1.0;
                    }
                    return -1.0;
                }

                if (n == 0){
                    return 1.0;
                }
                if (n == 1){
                    return x;
                }

                double P0 = 1.0;
                double P1 = x;
                double Pn;

                for(int k = 1; k < n; k++){
                    Pn = ((2 * k + 1) * x * P1 - (k * P0)) / (k + 1);
                    P0 = P1;
                    P1 = Pn;
                }

                return Pn;
            ''')


class Besseli0(SimpleCLLibrary):

    def __init__(self):
        """Return the zeroth-order modified Bessel function of the first kind

        Original author of C code: M.G.R. Vogelaar
        """
        cl_code = '''
            double y;
            
            if(fabs(x) < 3.75){
                y = (x / 3.75) * (x / 3.75);                  
                
                return 1.0 + y * (3.5156229 
                                  + y * (3.0899424
                                         + y * (1.2067492 
                                                + y * (0.2659732
                                                       + y * (0.360768e-1 
                                                              + y * 0.45813e-2)))));
            }
            
            y = 3.75 / fabs(x);
            return (exp(fabs(x)) / sqrt(fabs(x))) 
                    * (0.39894228
                       + y * (0.1328592e-1
                              + y * (0.225319e-2
                                     + y * (-0.157565e-2
                                            + y * (0.916281e-2
                                                   + y * (-0.2057706e-1
                                                          + y * (0.2635537e-1
                                                                 + y * (-0.1647633e-1
                                                                        + y * 0.392377e-2))))))));
        '''
        super(Besseli0, self).__init__('double', 'bessel_i0', [('double', 'x')], cl_code)


class LogBesseli0(SimpleCLLibrary):

    def __init__(self):
        """Return the log of the zeroth-order modified Bessel function of the first kind."""
        super(LogBesseli0, self).__init__(
            'double', 'log_bessel_i0', [('double', 'x')],
            '''
              if(x < 700){
                  return log(bessel_i0(x));
              }
              return x - log(2.0 * M_PI * x)/2.0;
            ''', dependency_list=(Besseli0(),))


class GammaCDF(SimpleCLLibraryFromFile):

    def __init__(self):
        """Calculate the Cumulative Distribution Function of the Gamma function.

        This computes: lower_incomplete_gamma(k, x/theta) / gamma(k)

        With k the shape parameter, theta the scale parameter, lower_incomplete_gamma the lower incomplete gamma
        function and gamma the complete gamma function.

        Function arguments:

         * shape: the shape parameter of the gamma distribution (often denoted :math:`k`)
         * scale: the scale parameter of the gamma distribution (often denoted :math:`\theta`)
        """
        super(GammaCDF, self).__init__(
            'double', 'gamma_cdf',
            [('double', 'shape'),
             ('double', 'scale'),
             ('double', 'x')],
            'return gamma_p(shape, x/scale);',
            dependency_list=(GammaP(),))


class GammaP(SimpleCLLibraryFromFile):

    def __init__(self):
        """Calculates the normalized/regularized lower incomplete gamma function returning values in the range [0, 1].
        """
        super(GammaP, self).__init__(
            'double', 'gamma_p', [('double', 'a'), ('double', 'x')],
            resource_filename('mot', 'data/opencl/gamma_p.cl'))


class LogCosh(SimpleCLLibrary):

    def __init__(self):
        """Computes :math:`log(cosh(x))`

        For large x this will try to estimate it without overflow. For small x we use the opencl functions log and cos.
        The estimation for large numbers has been taken from:
        https://github.com/JaneliaSciComp/tmt/blob/master/basics/logcosh.m
        """
        super(LogCosh, self).__init__(
            'double', 'log_cosh',
            [('double', 'x')],
            '''
                if(x < 50){
                    return log(cosh(x));
                }
                return fabs(x) + log(1 + exp(-2.0 * fabs(x))) - log(2.0);
            ''')


class Rand123(SimpleCLLibrary):

    def __init__(self):
        """Estimate various trigonometric functions additional to the OpenCL offerings."""
        super(Rand123, self).__init__('void', 'rand123', [], '', cl_extra=Rand123._get_random123_cl_code())

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
