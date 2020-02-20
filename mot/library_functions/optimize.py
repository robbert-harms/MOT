"""Several routines from the EISPACK-C code.

All routines are prefixed with 'eispack_' for use in MOT.

Reference:
    https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.html .
"""
from pkg_resources import resource_filename

from mot.lib.kernel_data import LocalMemory
from mot.library_functions import SimpleCLLibrary, SimpleCLLibraryFromFile

__author__ = 'Robbert Harms'
__date__ = '2019-12-17'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'


class bracket_spf(SimpleCLLibraryFromFile):

    def __init__(self, function_name):
        """The Bracket algorithm as a reusable library component.

        Since it is an ``_spf`` method, parts of the implementation are specialized for the given function name.
        Query the :meth:`get_cl_function_name` for the function call name.

        Args:
            function_name (str): the name of the evaluation function to call, defaults to 'evaluate'.
                This should point to a function with signature:

                    ``double evaluate(mot_float_type x, void* data_void);``
        """
        params = {
            'FUNCTION_NAME': function_name,
            'SPF_NAME': '_spf_' + function_name
        }

        super().__init__(
            'int', 'bracket' + params['SPF_NAME'], [],
            resource_filename('mot', 'data/opencl/bracket_spf.cl'),
            var_replace_dict=params)


class nmsimplex_spf(SimpleCLLibraryFromFile):

    def __init__(self, function_name):
        """The NMSimplex algorithm as a specialized function object.

        Since it is an ``_spf`` method, parts of the implementation are specialized for the given function name.
        Query the :meth:`get_cl_function_name` for the function call name.

        Args:
            function_name (str): the name of the evaluation function to call, defaults to 'evaluate'.
                This should point to a function with signature:

                    ``double evaluate(local mot_float_type* x, void* data_void);``
        """
        params = {
            'FUNCTION_NAME': function_name,
            'SPF_NAME': '_spf_' + function_name
        }

        super().__init__(
            'int', 'nmsimplex' + params['SPF_NAME'], [],
            resource_filename('mot', 'data/opencl/nmsimplex_spf.cl'),
            var_replace_dict=params)


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

        bracket_func = bracket_spf('powell_linear_eval_function')

        params = {
            'FUNCTION_NAME': eval_func.get_cl_function_name(),
            'NMR_PARAMS': nmr_parameters,
            'RESET_METHOD': reset_method.upper(),
            'PATIENCE': patience,
            'PATIENCE_LINE_SEARCH': patience if patience_line_search is None else patience_line_search,
            'BRACKET_FUNC': bracket_func.get_cl_code(),
            'BRACKET_FUNC_NAME': bracket_func.get_cl_function_name()
        }
        super().__init__(
            'int', 'powell', [
                'local mot_float_type* model_parameters',
                'void* data',
                'local mot_float_type* scratch_mot_float_type'
            ],
            resource_filename('mot', 'data/opencl/powell.cl'),
            var_replace_dict=params, **kwargs)

    def get_kernel_data(self):
        """Get the kernel data needed for this optimization routine to work."""
        return {
            'scratch_mot_float_type': LocalMemory(
                'mot_float_type',  3 * self._var_replace_dict['NMR_PARAMS'] + self._var_replace_dict['NMR_PARAMS']**2)
        }


class NMSimplex(SimpleCLLibrary):

    def __init__(self, function_name, nmr_parameters, patience=200, alpha=1.0, beta=0.5,
                 gamma=2.0, delta=0.5, scale=0.1, adaptive_scales=True, **kwargs):

        self._nmr_parameters = nmr_parameters

        simplex_func = nmsimplex_spf(function_name)

        if 'dependencies' in kwargs:
            kwargs['dependencies'] = list(kwargs['dependencies']) + [simplex_func]
        else:
            kwargs['dependencies'] = [simplex_func]

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

        super().__init__(('''
            int nmsimplex(local mot_float_type* model_parameters, void* data,
                          local mot_float_type* initial_simplex_scale,
                          local mot_float_type* nmsimplex_scratch){

                if(get_local_id(0) == 0){
                    %(INITIAL_SIMPLEX_SCALES)s
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                mot_float_type fdiff;
                mot_float_type psi = 0;

                return ''' + simplex_func.get_cl_function_name() + '''(
                    %(NMR_PARAMS)r, model_parameters, data, initial_simplex_scale,
                    &fdiff, psi, (int)(%(PATIENCE)r * (%(NMR_PARAMS)r+1)),
                    %(ALPHA)r, %(BETA)r, %(GAMMA)r, %(DELTA)r,
                    nmsimplex_scratch);
            }
        ''') % params, **kwargs)

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
        """The Subplex optimization routines."""
        dependencies = list(kwargs.get('dependencies', []))
        dependencies.append(eval_func)

        simplex_func = nmsimplex_spf('subspace_evaluate')
        dependencies.append(simplex_func)

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
            'MAX_SUBSPACE_LENGTH': (min(5, nmr_parameters) if max_subspace_length == 'auto' else max_subspace_length),
            'SIMPLEX_SPF': simplex_func.get_cl_function_name()
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

    def __init__(self, eval_func, nmr_parameters, nmr_observations, jacobian_func, patience=250,
                 step_bound=100.0, scale_diag=1, usertol_mult=30, **kwargs):
        """The Powell CL implementation.

        Args:
            eval_func (mot.lib.cl_function.CLFunction): the function we want to optimize, Should be of signature:
                ``void evaluate(local mot_float_type* x, void* data_void, local mot_float_type* result);``
            nmr_parameters (int): the number of parameters in the model, this will be hardcoded in the method
            nmr_observations (int): the number of observations in the model
            jacobian_func (mot.lib.cl_function.CLFunction): the function used to compute the Jacobian.
            patience (int): the patience of the Powell algorithm
            patience_line_search (int): the patience of the line search algorithm
            reset_method (str): one of ``RESET_TO_IDENTITY`` or ``EXTRAPOLATED_POINT``. The method used to
                reset the search directions every iteration.
        """
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
            'int', 'lmmin', ['local mot_float_type* const model_parameters',
                             'void* data',
                             'mot_float_type* scratch_mot_float_type',
                             'int* scratch_int'],
            resource_filename('mot', 'data/opencl/lmmin.cl'),
            var_replace_dict=var_replace_dict, **kwargs)

    def get_kernel_data(self):
        """Get the kernel data needed for this optimization routine to work."""
        return {
            'scratch_mot_float_type': LocalMemory(
                'mot_float_type', 8 +
                                  2 * self._var_replace_dict['NMR_OBSERVATIONS'] +
                                  5 * self._var_replace_dict['NMR_PARAMS'] +
                                  self._var_replace_dict['NMR_PARAMS'] * self._var_replace_dict['NMR_OBSERVATIONS']),
            'scratch_int': LocalMemory('int', self._var_replace_dict['NMR_PARAMS'])
        }
