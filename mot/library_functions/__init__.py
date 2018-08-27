import os
from mot.library_functions.base import SimpleCLLibrary, SimpleCLLibraryFromFile, CLLibrary
from pkg_resources import resource_filename
from mot.library_functions.unity import log1pmx
from mot.library_functions.polynomials import p1evl, polevl, ratevl, solve_cubic_pol_real
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
        super().__init__('''
            double firstLegendreTerm(double x, int n){
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
            }
        ''')


class Besseli0(SimpleCLLibrary):
    def __init__(self):
        """Return the zeroth-order modified Bessel function of the first kind

        Original author of C code: M.G.R. Vogelaar
        """
        super().__init__('''
            double bessel_i0(double x){
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
            }
        ''')


class LogBesseli0(SimpleCLLibrary):
    def __init__(self):
        """Return the log of the zeroth-order modified Bessel function of the first kind."""
        super().__init__('''
            double log_bessel_i0(double x){
                if(x < 700){
                  return log(bessel_i0(x));
                }
                return x - log(2.0 * M_PI * x)/2.0;
            }
        ''', dependencies=(Besseli0(),))


class LogCosh(SimpleCLLibrary):
    def __init__(self):
        """Computes :math:`log(cosh(x))`

        For large x this will try to estimate it without overflow. For small x we use the opencl functions log and cos.
        The estimation for large numbers has been taken from:
        https://github.com/JaneliaSciComp/tmt/blob/master/basics/logcosh.m
        """
        super().__init__('''
            double log_cosh(double x){
                if(x < 50){
                    return log(cosh(x));
                }
                return fabs(x) + log(1 + exp(-2.0 * fabs(x))) - log(2.0);
            }
        ''')


class Rand123(SimpleCLLibrary):
    def __init__(self):
        generator = 'threefry'

        src = open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/openclfeatures.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/array.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/{}.h'.format(generator)), ),
                    'r').read()
        src += (open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/rand123.h'), ), 'r').read() % {
            'GENERATOR_NAME': (generator)
        })

        super().__init__('void rand123(){}', cl_extra=src)


class EuclidianNormFunction(SimpleCLLibraryFromFile):
    def __init__(self, memspace='private', memtype='mot_float_type'):
        """A CL functions for calculating the Euclidian distance between n values.

        Args:
            memspace (str): The memory space of the memtyped array (private, constant, global).
            memtype (str): the memory type to use, double, float, mot_float_type, ...
        """
        super().__init__(
            memtype,
            self.__class__.__name__ + '_' + memspace + '_' + memtype,
            [],
            resource_filename('mot', 'data/opencl/euclidian_norm.cl'),
            var_replace_dict={'MEMSPACE': memspace, 'MEMTYPE': memtype})


class LibNMSimplex(SimpleCLLibraryFromFile):

    def __init__(self, function_name):
        """The NMSimplex algorithm as a reusable library component.

        Args:
            function_name (str): the name of the evaluation function to call, defaults to 'evaluate'.
                This should point to a function with signature:

                    ``double evaluate(local mot_float_type* x, void* data_void);``
        """
        params = {
            'FUNCTION_NAME': function_name
        }

        super().__init__(
            'int', 'lib_nmsimplex', [],
            resource_filename('mot', 'data/opencl/lib_nmsimplex.cl'),
            var_replace_dict=params)


class NMSimplex(SimpleCLLibrary):

    def __init__(self, function_name, nmr_parameters, patience=200, alpha=1.0, beta=0.5,
                 gamma=2.0, delta=0.5, scale=1.0, adaptive_scales=True, **kwargs):

        if 'dependencies' in kwargs:
            kwargs['dependencies'] = list(kwargs['dependencies']) + [LibNMSimplex(function_name)]
        else:
            kwargs['dependencies'] = [LibNMSimplex(function_name)]

        params = {'NMR_PARAMS': nmr_parameters,
                  'PATIENCE': patience,
                  'ALPHA': alpha,
                  'BETA': beta,
                  'GAMMA': gamma,
                  'DELTA': delta,
                  'INITIAL_SIMPLEX_SCALES': '\n'.join('initial_simplex_scale[{}] = {};'.format(ind, scale)
                                                      for ind in range(nmr_parameters))}

        if adaptive_scales:
            params.update(
                {'ALPHA': 1,
                 'BETA': 0.75 - 1.0 / (2 * nmr_parameters),
                 'GAMMA': 1 + 2.0 / nmr_parameters,
                 'DELTA': 1 - 1.0 / nmr_parameters}
            )

        super().__init__('''
            int nmsimplex(local mot_float_type* model_parameters, void* data){
                local mot_float_type initial_simplex_scale[%(NMR_PARAMS)r];
                
                if(get_local_id(0) == 0){
                    %(INITIAL_SIMPLEX_SCALES)s
                }
                barrier(CLK_LOCAL_MEM_FENCE);
    
                mot_float_type fdiff;
                mot_float_type psi = 0;
                local mot_float_type nmsimplex_scratch[
                    %(NMR_PARAMS)r * 2 + (%(NMR_PARAMS)r + 1) * (%(NMR_PARAMS)r + 1)];
    
                return lib_nmsimplex(%(NMR_PARAMS)r, model_parameters, data, initial_simplex_scale,
                                     &fdiff, psi, (int)(%(PATIENCE)r * (%(NMR_PARAMS)r+1)),
                                     %(ALPHA)r, %(BETA)r, %(GAMMA)r, %(DELTA)r,
                                     nmsimplex_scratch);
            }
        ''' % params, **kwargs)


class Powell(SimpleCLLibraryFromFile):

    def __init__(self, function_name, nmr_parameters, patience=2, patience_line_search=None,
                 reset_method='EXTRAPOLATED_POINT', **kwargs):
        """The Powell CL implementation.

        Args:
            function_name (str): the name of the function we want to optimize, this will be hardcoded in the
                powell method. Should be of signature: ``double evaluate(local mot_float_type* x, void* data_void);``
            nmr_parameters (int): the number of parameters in the model, this will be hardcoded in the method
            patience (int): the patience of the Powell algorithm
            patience_line_search (int): the patience of the line search algorithm. If None, we set it equal to the
                patience.
            reset_method (str): one of ``RESET_TO_IDENTITY`` or ``EXTRAPOLATED_POINT``. The method used to
                reset the search directions every iteration.
        """
        params = {
            'FUNCTION_NAME': function_name,
            'NMR_PARAMS': nmr_parameters,
            'RESET_METHOD': reset_method.upper(),
            'PATIENCE': patience,
            'PATIENCE_LINE_SEARCH': patience if patience_line_search is None else patience_line_search
        }
        super().__init__(
            'int', 'powell', [('local mot_float_type*', 'model_parameters'), ('void*', 'data')],
            resource_filename('mot', 'data/opencl/powell.cl'),
            var_replace_dict=params, **kwargs)


class Subplex(SimpleCLLibraryFromFile):

    def __init__(self, function_name, nmr_parameters, patience=10,
                 patience_nmsimplex=100, alpha=1.0, beta=0.5, gamma=2.0, delta=0.5, scale=1.0, psi=0.001, omega=0.01,
                 adaptive_scales=True, min_subspace_length='auto', max_subspace_length='auto', **kwargs):
        """The Subplex optimization routines.

        Args:
            function_name (str): the name of the function we want to optimize, this will be hardcoded in the
                powell method. Should be of signature: ``double evaluate(local mot_float_type* x, void* data_void);``
            nmr_parameters (int): the number of parameters in the model, this will be hardcoded in the method
            patience (int): the patience of the Powell algorithm
            patience_nmsimplex (int): the patience of the Nelder-Mead simplex routine
            scale (double): the scale of the initial simplex, default 1.0
            alpha (double): reflection coefficient, default 1.0
            beta (double): contraction coefficient, default 0.5
            gamma (double); expansion coefficient, default 2.0
            delta (double); shrinkage coefficient, default 0.5
            psi (double): subplex specific, simplex reduction coefficient, default 0.001.
            omega (double): subplex specific, scaling reduction coefficient, default 0.01
            min_subspace_length (int): the minimum subspace length, defaults to min(2, n).
                This should hold: (1 <= min_s_d <= max_s_d <= n and min_s_d*ceil(n/nsmax_s_dmax) <= n)
            max_subspace_length (int): the maximum subspace length, defaults to min(5, n).
                This should hold: (1 <= min_s_d <= max_s_d <= n and min_s_d*ceil(n/max_s_d) <= n)

            adaptive_scales (boolean): if set to True we use adaptive scales instead of the default scale values.
                This sets the scales to:

                .. code-block:: python

                    n = <# parameters>

                    alpha = 1
                    beta  = 0.75 - 1.0 / (2 * n)
                    gamma = 1 + 2.0 / n
                    delta = 1 - 1.0 / n
        """
        if 'dependencies' in kwargs:
            kwargs['dependencies'] = list(kwargs['dependencies']) + [LibNMSimplex('subspace_evaluate')]
        else:
            kwargs['dependencies'] = [LibNMSimplex('subspace_evaluate')]

        params = {
            'FUNCTION_NAME': function_name,
            'PATIENCE': patience,
            'PATIENCE_NMSIMPLEX': patience_nmsimplex,
            'ALPHA': alpha,
            'BETA': beta,
            'GAMMA': gamma,
            'DELTA': delta,
            'PSI': psi,
            'OMEGA': omega,
            'NMR_PARAMS': nmr_parameters,
            'ADAPTIVE_SCALES': int(bool(adaptive_scales)),
            'MIN_SUBSPACE_LENGTH': (min(2, nmr_parameters) if min_subspace_length == 'auto' else min_subspace_length),
            'MAX_SUBSPACE_LENGTH': (min(5, nmr_parameters) if max_subspace_length == 'auto' else max_subspace_length)
        }

        s = ''
        for ind in range(nmr_parameters):
            s += 'initial_simplex_scale[{}] = {};'.format(ind, scale)
        params['INITIAL_SIMPLEX_SCALES'] = s

        super().__init__(
            'int', 'subplex', [('local mot_float_type* const', 'model_parameters'), ('void*', 'data')],
            resource_filename('mot', 'data/opencl/subplex.cl'), var_replace_dict=params, **kwargs)


class LevenbergMarquardt(SimpleCLLibraryFromFile):

    def __init__(self, function_name, nmr_parameters, nmr_observations, patience=250, step_bound=100.0, scale_diag=1,
                 usertol_mult=30, **kwargs):
        """The Powell CL implementation.

        Args:
            function_name (str): the name of the function we want to optimize, this will be hardcoded in the
                powell method. Should be of signature:
                ``void evaluate(local mot_float_type* x, void* data_void, local mot_float_type* result);``
            nmr_parameters (int): the number of parameters in the model, this will be hardcoded in the method
            patience (int): the patience of the Powell algorithm
            patience_line_search (int): the patience of the line search algorithm
            reset_method (str): one of ``RESET_TO_IDENTITY`` or ``EXTRAPOLATED_POINT``. The method used to
                reset the search directions every iteration.
        """
        params = {
            'FUNCTION_NAME': function_name,
            'NMR_PARAMS': nmr_parameters,
            'PATIENCE': patience,
            'NMR_OBSERVATIONS': nmr_observations,
            'SCALE_DIAG': int(bool(scale_diag)),
            'STEP_BOUND': step_bound,
            'USERTOL_MULT': usertol_mult
        }

        super().__init__(
            'int', 'lmmin', [('local mot_float_type* const', 'model_parameters'),
                             ('void*', 'data'),
                             ('global mot_float_type*', 'fjac')],
            resource_filename('mot', 'data/opencl/lmmin.cl'),
            var_replace_dict=params, **kwargs)
