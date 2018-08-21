from mot.lib.cl_function import SimpleCLFunction
from mot.configuration import CLRuntimeInfo
from mot.lib.kernel_data import Array, Zeros
from mot.library_functions import Powell, Subplex, NMSimplex, LevenbergMarquardt
from mot.optimize.base import OptimizeResults

__author__ = 'Robbert Harms'
__date__ = '2018-08-01'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


def minimize(func, x0, data=None, method=None, nmr_observations=None, cl_runtime_info=None, options=None):
    """Minimization of scalar function of one or more variables.

    Args:
        func (mot.lib.cl_function.CLFunction): A CL function with the signature:

            .. code-block:: c

                double <func_name>(local const mot_float_type* const x,
                                   void* data,
                                   local mot_float_type* objective_list);

            The objective list needs to be filled when the provided pointer is not null. It should contain
            the function values for each observation. This list is used by non-linear least-squares routines,
            and will be squared by the least-square optimizer. This is only used by the ``Levenberg-Marquardt`` routine.

        x0 (ndarray): Initial guess. Array of real elements of size (n, p), for 'n' problems and 'p'
            independent variables.
        data (mot.lib.kernel_data.KernelData): the kernel data we will load. This is returned to the likelihood function
            as the ``void* data`` pointer.
        method (str): Type of solver.  Should be one of:
            - 'Levenberg-Marquardt'
            - 'Nelder-Mead'
            - 'Powell'
            - 'Subplex'

            If not given, defaults to 'Powell'.

        nmr_observations (int): the number of observations returned by the optimization function.
            This is only needed for the ``Levenberg-Marquardt`` method.
        cl_runtime_info (mot.configuration.CLRuntimeInfo): the CL runtime information
        options (dict): A dictionary of solver options. All methods accept the following generic options:
                patience (int): Maximum number of iterations to perform.

    Returns:
        mot.optimize.base.OptimizeResults:
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array.
    """
    if not method:
        method = 'Powell'

    cl_runtime_info = cl_runtime_info or CLRuntimeInfo()

    if len(x0.shape) < 2:
        x0 = x0[..., None]

    if method == 'Powell':
        return _minimize_powell(func, x0, cl_runtime_info, data, options)
    elif method == 'Nelder-Mead':
        return _minimize_nmsimplex(func, x0, cl_runtime_info, data, options)
    elif method == 'Levenberg-Marquardt':
        return _minimize_levenberg_marquardt(func, x0, nmr_observations, cl_runtime_info, data, options)
    elif method == 'Subplex':
        return _minimize_subplex(func, x0, cl_runtime_info, data, options)
    raise ValueError('Could not find the specified method "{}".'.format(method))


def get_minimizer_options(method):
    """Return a dictionary with the default options for the given minimization method.

    Args:
        method (str): the name of the method we want the options off

    Returns:
        dict: a dictionary with the default options
    """
    if method == 'Powell':
        return {'patience': 2,
                'patience_line_search': None,
                'reset_method': 'EXTRAPOLATED_POINT'}

    elif method == 'Nelder-Mead':
        return {'patience': 200,
                'alpha': 1.0, 'beta': 0.5, 'gamma': 2.0, 'delta': 0.5, 'scale': 1.0,
                'adaptive_scales': True}

    elif method == 'Levenberg-Marquardt':
        return {'patience': 250, 'step_bound': 100.0, 'scale_diag': 1, 'usertol_mult': 30}

    elif method == 'Subplex':
        return {'patience': 10,
                'patience_nmsimplex': 100,
                'alpha': 1.0, 'beta': 0.5, 'gamma': 2.0, 'delta': 0.5, 'scale': 1.0, 'psi': 0.001, 'omega': 0.01,
                'adaptive_scales': True,
                'min_subspace_length': 'auto',
                'max_subspace_length': 'auto'}

    raise ValueError('Could not find the specified method "{}".'.format(method))


def _clean_options(method, provided_options):
    """Clean the given input options.

    This will make sure that all options are present, either with their default values or with the given values,
    and that no other options are present then those supported.

    Args:
        method (str): the method name
        provided_options (dict): the given options

    Returns:
        dict: the resulting options dictionary
    """
    provided_options = provided_options or {}
    default_options = get_minimizer_options(method)

    result = {}

    for name, default in default_options.items():
        if name in provided_options:
            result[name] = provided_options[name]
        else:
            result[name] = default_options[name]
    return result


def _minimize_powell(func, x0, cl_runtime_info, data=None, options=None):
    """
    Options:
        patience (int): Used to set the maximum number of iterations to patience*(number_of_parameters+1)
        reset_method (str): one of 'EXTRAPOLATED_POINT' or 'RESET_TO_IDENTITY' lower case or upper case.
        patience_line_search (int): the patience of the searching algorithm. Defaults to the
            same patience as for the Powell algorithm itself.
    """
    options = _clean_options('Powell', options)

    nmr_problems = x0.shape[0]
    nmr_parameters = x0.shape[1]

    kernel_data = {'model_parameters': Array(x0, ctype='mot_float_type', mode='rw'),
                   'data': data}

    eval_func = SimpleCLFunction.from_string('''
        double evaluate(local mot_float_type* x, void* data){
            return ''' + func.get_cl_function_name() + '''(x, data, 0);
        }
    ''', dependencies=[func])

    optimizer_func = Powell('evaluate', nmr_parameters, dependencies=[eval_func], **options)

    return_code = optimizer_func.evaluate(
        kernel_data, nmr_problems,
        use_local_reduction=all(env.is_gpu for env in cl_runtime_info.get_cl_environments()),
        cl_runtime_info=cl_runtime_info)

    return OptimizeResults({'x': kernel_data['model_parameters'].get_data(),
                            'status': return_code})


def _minimize_nmsimplex(func, x0, cl_runtime_info, data=None, options=None):
    """Use the Nelder-Mead simplex method to calculate the optimimum.

    The scales should satisfy the following constraints:

        .. code-block:: python

            alpha > 0
            0 < beta < 1
            gamma > 1
            gamma > alpha
            0 < delta < 1

    Options:
        patience (int): Used to set the maximum number of iterations to patience*(number_of_parameters+1)
        scale (double): the scale of the initial simplex, default 1.0
        alpha (double): reflection coefficient, default 1.0
        beta (double): contraction coefficient, default 0.5
        gamma (double); expansion coefficient, default 2.0
        delta (double); shrinkage coefficient, default 0.5
        adaptive_scales (boolean): if set to True we use adaptive scales instead of the default scale values.
            This sets the scales to:

            .. code-block:: python

                n = <# parameters>

                alpha = 1
                beta  = 0.75 - 1.0 / (2 * n)
                gamma = 1 + 2.0 / n
                delta = 1 - 1.0 / n

            Following the paper [1]

    References:
        [1] Gao F, Han L. Implementing the Nelder-Mead simplex algorithm with adaptive parameters.
              Comput Optim Appl. 2012;51(1):259-277. doi:10.1007/s10589-010-9329-3.
    """
    options = _clean_options('Nelder-Mead', options)

    nmr_problems = x0.shape[0]
    nmr_parameters = x0.shape[1]

    kernel_data = {'model_parameters': Array(x0, ctype='mot_float_type', mode='rw'),
                   'data': data}

    eval_func = SimpleCLFunction.from_string('''
        double evaluate(local mot_float_type* x, void* data){
            return ''' + func.get_cl_function_name() + '''(x, data, 0);
        }
    ''', dependencies=[func])

    optimizer_func = NMSimplex('evaluate', nmr_parameters, dependencies=[eval_func], **options)

    return_code = optimizer_func.evaluate(
        kernel_data, nmr_problems,
        use_local_reduction=all(env.is_gpu for env in cl_runtime_info.get_cl_environments()),
        cl_runtime_info=cl_runtime_info)

    return OptimizeResults({'x': kernel_data['model_parameters'].get_data(),
                            'status': return_code})


def _minimize_subplex(func, x0, cl_runtime_info, data=None, options=None):
    """Variation on the Nelder-Mead Simplex method by Thomas H. Rowan.

    This method uses NMSimplex to search subspace regions for the minimum. See Rowan's thesis titled
    "Functional Stability analysis of numerical algorithms" for more details.

     The scales should satisfy the following constraints:

        .. code-block:: python

            alpha > 0
            0 < beta < 1
            gamma > 1
            gamma > alpha
            0 < delta < 1

    Options:
        patience (int): Used to set the maximum number of iterations to patience*(number_of_parameters+1)
        patience_nmsimplex (int): The maximum patience for each subspace search.
            For each subspace search we set the number of iterations to patience*(number_of_parameters_subspace+1)
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

    References:
        [1] Gao F, Han L. Implementing the Nelder-Mead simplex algorithm with adaptive parameters.
              Comput Optim Appl. 2012;51(1):259-277. doi:10.1007/s10589-010-9329-3.
    """
    options = _clean_options('Subplex', options)

    nmr_problems = x0.shape[0]
    nmr_parameters = x0.shape[1]

    kernel_data = {'model_parameters': Array(x0, ctype='mot_float_type', mode='rw'),
                   'data': data}

    eval_func = SimpleCLFunction.from_string('''
        double evaluate(local mot_float_type* x, void* data){
            return ''' + func.get_cl_function_name() + '''(x, data, 0);
        }
    ''', dependencies=[func])

    optimizer_func = Subplex('evaluate', nmr_parameters, dependencies=[eval_func], **options)

    return_code = optimizer_func.evaluate(
        kernel_data, nmr_problems,
        use_local_reduction=all(env.is_gpu for env in cl_runtime_info.get_cl_environments()),
        cl_runtime_info=cl_runtime_info)

    return OptimizeResults({'x': kernel_data['model_parameters'].get_data(),
                            'status': return_code})


def _minimize_levenberg_marquardt(func, x0, nmr_observations, cl_runtime_info, data=None, options=None):
    options = _clean_options('Levenberg-Marquardt', options)

    nmr_problems = x0.shape[0]
    nmr_parameters = x0.shape[1]

    if nmr_observations < x0.shape[1]:
        raise ValueError('The number of instances per problem must be greater than the number of parameters')

    kernel_data = {'model_parameters': Array(x0, ctype='mot_float_type', mode='rw'),
                   'data': data,
                   'fjac': Zeros((nmr_problems, nmr_parameters, nmr_observations), ctype='mot_float_type',
                                 mode='rw')}

    eval_func = SimpleCLFunction.from_string('''
        void evaluate(local mot_float_type* x, void* data, local mot_float_type* result){
            ''' + func.get_cl_function_name() + '''(x, data, result);
        }
    ''', dependencies=[func])

    optimizer_func = LevenbergMarquardt('evaluate', nmr_parameters, nmr_observations,
                                        dependencies=[eval_func], **options)

    return_code = optimizer_func.evaluate(
        kernel_data, nmr_problems,
        use_local_reduction=all(env.is_gpu for env in cl_runtime_info.get_cl_environments()),
        cl_runtime_info=cl_runtime_info)

    return OptimizeResults({'x': kernel_data['model_parameters'].get_data(),
                            'status': return_code})
