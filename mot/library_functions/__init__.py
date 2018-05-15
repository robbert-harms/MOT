import os
from mot.library_functions.base import SimpleCLLibrary, SimpleCLLibraryFromFile
from pkg_resources import resource_filename
from mot.library_functions.unity import log1pmx
from mot.library_functions.polynomial_evaluations import p1evl, polevl, ratevl
from mot.library_functions.continuous_distributions.normal import normal_cdf, normal_pdf, normal_ppf
from mot.library_functions.continuous_distributions.gamma import gamma_pdf, gamma_ppf, gamma_cdf
from mot.library_functions.error_functions import dawson, CerfImWOfX, erfi

__author__ = 'Robbert Harms'
__date__ = '2018-05-07'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class FirstLegendreTerm(SimpleCLLibrary):
    def __init__(self):
        """Compute the first term of the legendre polynomial for the given value x and the polynomial degree n.

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
        super(Besseli0, self).__init__(
            'double', 'bessel_i0', [('double', 'x')],
            '''
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
            ''')


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
            ''', dependencies=(Besseli0(),))


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
        """The NMSimplex algorithm as a reusable library component.

        Args:
            evaluate_fname (str): the name of the evaluation function to call, defaults to 'evaluate'.
                This should point to a function with signature:

                    ``double evaluate(mot_float_type* x, void* data_void);``
        """
        params = {
            'EVALUATE_FNAME': str(evaluate_fname)
        }

        super(LibNMSimplex, self).__init__(
            'int', 'lib_nmsimplex', [],
            resource_filename('mot', 'data/opencl/lib_nmsimplex.cl'),
            var_replace_dict=params)
