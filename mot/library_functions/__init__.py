import os

from mot.lib.cl_function import SimpleCLFunction, SimpleCLCodeObject
from mot.lib.kernel_data import LocalMemory
from mot.library_functions.base import SimpleCLLibrary, SimpleCLLibraryFromFile, CLLibrary
from pkg_resources import resource_filename
from mot.library_functions.unity import log1pmx
from mot.library_functions.polynomials import p1evl, polevl, ratevl, solve_cubic_pol_real
from mot.library_functions.continuous_distributions.normal import normal_cdf, normal_pdf, normal_logpdf, normal_ppf
from mot.library_functions.continuous_distributions.gamma import gamma_pdf, gamma_logpdf, gamma_ppf, gamma_cdf
from mot.library_functions.error_functions import dawson, CerfImWOfX, erfi
from mot.library_functions.legendre_polynomial import FirstLegendreTerm, LegendreTerms, \
    EvenLegendreTerms, OddLegendreTerms


__author__ = 'Robbert Harms'
__date__ = '2018-05-07'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


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


class Rand123(SimpleCLCodeObject):
    def __init__(self):
        generator = 'threefry'

        src = open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/openclfeatures.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/array.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/{}.h'.format(generator)), ),
                    'r').read()
        src += (open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/rand123.h'), ), 'r').read() % {
            'GENERATOR_NAME': (generator)
        })
        super().__init__(src)


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


class simpsons_rule(SimpleCLLibrary):

    def __init__(self, function_name):
        """Create a CL function for integrating a function using Simpson's rule.

        This creates a CL function specifically meant for integrating the function of the given name.
        The name of the generated CL function will be 'simpsons_rule_<function_name>'.

        Args:
            function_name (str): the name of the function to integrate, accepting the arguments:
                - a: the lower bound of the integral
                - b: the upper bound of the integral
                - n: the number of steps, i.e. the number of approximations to make
                - data: a pointer to some data, this is passed on to the function we are integrating.
        """
        super().__init__('''
            double simpsons_rule_{f}(double a, double b, uint n, void* data){{
                double h = (b - a) / n;

                double sum_odds = {f}(a + h/2.0, data);
                double sum_evens = 0.0;

                double x;

                for(uint i = 1; i < n; i++){{
                    sum_odds += {f}(a + h * i + h / 2.0, data);
                    sum_evens += {f}(a + h * i, data);
                }}

                return h / 6.0 * ({f}(a, data) + {f}(b, data) + 4.0 * sum_odds + 2.0 * sum_evens);
            }}
        '''.format(f=function_name))


class linear_cubic_interpolation(SimpleCLLibrary):

    def __init__(self):
        """Cubic interpolation for a one-dimensional grid.

        This uses the theory of Cubic Hermite splines for interpolating a one-dimensional grid of values.

        At the borders, it will clip the values to the nearest border.

        For more information on this method, see https://en.wikipedia.org/wiki/Cubic_Hermite_spline.

        Example usage:
            constant float data[] = {1.0, 2.0, 5.0, 6.0};
            linear_cubic_interpolation(1.5, 4, data);
        """
        super().__init__('''
            double linear_cubic_interpolation(double x, int y_len, constant float* y_values){
                int n = x;
                double u = x - n;

                double p0 = y_values[min(max((int)0, n - 1), y_len - 1)];
                double p1 = y_values[min(max((int)0, n    ), y_len - 1)];
                double p2 = y_values[min(max((int)0, n + 1), y_len - 1)];
                double p3 = y_values[min(max((int)0, n + 2), y_len - 1)];

                double a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
                double b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
                double c = 0.5 * (-p0 + p2);
                double d = p1;

                return d + u * (c + u * (b + u * a));                
            }
        ''')


class Powell(SimpleCLLibraryFromFile):

    def __init__(self, eval_func, nmr_parameters, patience=2, patience_line_search=None,
                 reset_method='EXTRAPOLATED_POINT', **kwargs):
        """The Powell CL implementation.

        Args:
            eval_func (mot.lib.cl_function.CLFunction): the function we want to optimize, Should be of signature:
                ``double evaluate(local mot_float_type* x, void* data_void);``
            nmr_parameters (int): the number of parameters in the model, this will be hardcoded in the method
            patience (int): the patience of the Powell algorithm
            patience_line_search (int): the patience of the line search algorithm. If None, we set it equal to the
                patience.
            reset_method (str): one of ``RESET_TO_IDENTITY`` or ``EXTRAPOLATED_POINT``. The method used to
                reset the search directions every iteration.
        """
        dependencies = list(kwargs.get('dependencies', []))
        dependencies.append(eval_func)
        kwargs['dependencies'] = dependencies

        params = {
            'FUNCTION_NAME': eval_func.get_cl_function_name(),
            'NMR_PARAMS': nmr_parameters,
            'RESET_METHOD': reset_method.upper(),
            'PATIENCE': patience,
            'PATIENCE_LINE_SEARCH': patience if patience_line_search is None else patience_line_search
        }
        super().__init__(
            'int', 'powell', [
                'local mot_float_type* model_parameters',
                'void* data',
                'local mot_float_type* powell_scratch'
            ],
            resource_filename('mot', 'data/opencl/powell.cl'),
            var_replace_dict=params, **kwargs)

    def get_kernel_data(self):
        """Get the kernel data needed for this optimization routine to work."""
        return {
            'powell_scratch': LocalMemory(
                'mot_float_type',  self._var_replace_dict['NMR_PARAMS']
                                    + self._var_replace_dict['NMR_PARAMS']**2)
        }


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

        self._nmr_parameters = nmr_parameters

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
            int nmsimplex(local mot_float_type* model_parameters, void* data, 
                          local mot_float_type* initial_simplex_scale, 
                          local mot_float_type* nmsimplex_scratch){

                if(get_local_id(0) == 0){
                    %(INITIAL_SIMPLEX_SCALES)s
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                mot_float_type fdiff;
                mot_float_type psi = 0;

                return lib_nmsimplex(%(NMR_PARAMS)r, model_parameters, data, initial_simplex_scale,
                                     &fdiff, psi, (int)(%(PATIENCE)r * (%(NMR_PARAMS)r+1)),
                                     %(ALPHA)r, %(BETA)r, %(GAMMA)r, %(DELTA)r,
                                     nmsimplex_scratch);
            }
        ''' % params, **kwargs)

    def get_kernel_data(self):
        """Get the kernel data needed for this optimization routine to work."""
        return {
            'nmsimplex_scratch': LocalMemory(
                'mot_float_type', self._nmr_parameters * 2 + (self._nmr_parameters + 1) ** 2 + 1),
            'initial_simplex_scale': LocalMemory('mot_float_type', self._nmr_parameters)
        }


class Subplex(SimpleCLLibraryFromFile):

    def __init__(self, eval_func, nmr_parameters, patience=10,
                 patience_nmsimplex=100, alpha=1.0, beta=0.5, gamma=2.0, delta=0.5, scale=1.0, psi=0.001, omega=0.01,
                 adaptive_scales=True, min_subspace_length='auto', max_subspace_length='auto', **kwargs):
        """The Subplex optimization routines.

        Args:
            eval_func (mot.lib.cl_function.CLFunction): the function we want to optimize, Should be of signature:
                ``double evaluate(local mot_float_type* x, void* data_void);``
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
        dependencies = list(kwargs.get('dependencies', []))
        dependencies.append(eval_func)
        dependencies.append(LibNMSimplex('subspace_evaluate'))
        kwargs['dependencies'] = dependencies

        params = {
            'FUNCTION_NAME': eval_func.get_cl_function_name(),
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
            'int', 'subplex', [
                'local mot_float_type* const model_parameters',
                'void* data',
                'local mot_float_type* initial_simplex_scale',
                'local mot_float_type* subplex_scratch_float',
                'local int* subplex_scratch_int'
            ],
            resource_filename('mot', 'data/opencl/subplex.cl'), var_replace_dict=params, **kwargs)

    def get_kernel_data(self):
        """Get the kernel data needed for this optimization routine to work."""
        return {
            'subplex_scratch_float': LocalMemory(
                'mot_float_type', 4 + self._var_replace_dict['NMR_PARAMS'] * 2
                                    + self._var_replace_dict['MAX_SUBSPACE_LENGTH'] * 2
                                    + (self._var_replace_dict['MAX_SUBSPACE_LENGTH'] * 2
                                       + self._var_replace_dict['MAX_SUBSPACE_LENGTH']+1)**2 + 1),
            'subplex_scratch_int': LocalMemory(
                'int', 2 + self._var_replace_dict['NMR_PARAMS']
                         + (self._var_replace_dict['NMR_PARAMS'] // self._var_replace_dict['MIN_SUBSPACE_LENGTH'])),
            'initial_simplex_scale': LocalMemory('mot_float_type', self._var_replace_dict['NMR_PARAMS'])
        }


class LevenbergMarquardt(SimpleCLLibraryFromFile):

    def __init__(self, eval_func, nmr_parameters, nmr_observations, patience=250,
                 step_bound=100.0, scale_diag=1, usertol_mult=30, jacobian_func=None, **kwargs):
        """The Powell CL implementation.

        Args:
            eval_func (mot.lib.cl_function.CLFunction): the function we want to optimize, Should be of signature:
                ``void evaluate(local mot_float_type* x, void* data_void, local mot_float_type* result);``
            nmr_parameters (int): the number of parameters in the model, this will be hardcoded in the method
            patience (int): the patience of the Powell algorithm
            patience_line_search (int): the patience of the line search algorithm
            reset_method (str): one of ``RESET_TO_IDENTITY`` or ``EXTRAPOLATED_POINT``. The method used to
                reset the search directions every iteration.
            jacobian_func (mot.lib.cl_function.CLFunction or None): the function used to compute the Jacobian.
                If not given, we will use a numerical differentiation
        """
        if not jacobian_func:
            jacobian_func = self._get_numerical_jacobian_func(eval_func.get_cl_function_name(),
                                                              nmr_parameters, nmr_observations)
        dependencies = list(kwargs.get('dependencies', []))
        dependencies.append(eval_func)
        dependencies.append(jacobian_func)
        kwargs['dependencies'] = dependencies

        var_replace_dict = {
            'FUNCTION_NAME': eval_func.get_cl_function_name(),
            'JACOBIAN_FUNCTION_NAME': jacobian_func.get_cl_function_name(),
            'NMR_PARAMS': nmr_parameters,
            'PATIENCE': patience,
            'NMR_OBSERVATIONS': nmr_observations,
            'SCALE_DIAG': int(bool(scale_diag)),
            'STEP_BOUND': step_bound,
            'USERTOL_MULT': usertol_mult
        }

        super().__init__(
            'int', 'lmmin', [('local mot_float_type* const', 'model_parameters'),
                             ('void*', 'data')],
            resource_filename('mot', 'data/opencl/lmmin.cl'),
            var_replace_dict=var_replace_dict, **kwargs)

    def _get_numerical_jacobian_func(self, function_name, nmr_params, nmr_observations):
        return SimpleCLFunction.from_string(r'''
            void compute_jacobian(local mot_float_type* model_parameters,
                                  void* data,
                                  local mot_float_type* fvec,
                                  local mot_float_type* const fjac){
                /**
                 * Compute the Jacobian for use in the LM method.
                 *
                 * This should place the output in the ``fjac`` matrix.
                 *
                 * Parameters:
                 *
                 *   model_parameters: (nmr_params,) the current point around which we want to know the Jacobian
                 *   data: the current modeling data, used by the objective function
                 *   fvec: (nmr_observations,), the function values corresponding to the current model parameters
                 *   fjac: (nmr_parameters, nmr_observations), the memory location for the Jacobian
                 */
                int i, j;
                local mot_float_type temp, step;
                
                mot_float_type EPS = 30 * MOT_EPSILON;
                
                for (j = 0; j < %(NMR_PARAMS)s; j++) {
                    if(get_local_id(0) == 0){
                        temp = model_parameters[j];
                        step = max(EPS*EPS, EPS * fabs(temp));
                        model_parameters[j] += step; /* replace temporarily */
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
    
                    %(FUNCTION_NAME)s(model_parameters, data, fjac + j*%(NMR_OBSERVATIONS)s);
    
                    if(get_local_id(0) == 0){
                        for (i = 0; i < %(NMR_OBSERVATIONS)s; i++){
                            fjac[j*%(NMR_OBSERVATIONS)s+i] = (fjac[j*%(NMR_OBSERVATIONS)s+i] - fvec[i]) / step;
                        }
                        model_parameters[j] = temp; /* restore */
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
        ''' % dict(FUNCTION_NAME=function_name, NMR_PARAMS=nmr_params, NMR_OBSERVATIONS=nmr_observations))
