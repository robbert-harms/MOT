import os

from pkg_resources import resource_filename

from mot.lib.cl_function import SimpleCLFunction
from mot.lib.cl_runtime_info import CLRuntimeInfo
from mot.lib.kernel_data import Array, Zeros, LocalMemory
from mot.library_functions import LibNMSimplex
from mot.optimize.base import OptimizeResults

__author__ = 'Robbert Harms'
__date__ = '2018-08-01'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


def minimize(model, x0, method=None, cl_runtime_info=None, options=None):
    """Minimization of scalar function of one or more variables.

    Args:
        model (mot.lib.model_interfaces.OptimizeModelInterface): the model we want to optimize
        x0 (ndarray): Initial guess. Array of real elements of size (n, p), for 'n' problems and 'p' independent variables.
        method (str): Type of solver.  Should be one of:
            - 'Levenberg-Marquardt'
            - 'Nelder-Mead'
            - 'Powell'
            - 'Subplex'

            If not given, defaults to 'Powell'.

        cl_runtime_info (mot.lib.cl_runtime_info.CLRuntimeInfo): the CL runtime information
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
        return _minimize_powell(model, x0, cl_runtime_info, options)
    elif method == 'Nelder-Mead':
        return _minimize_nmsimplex(model, x0, cl_runtime_info, options)
    elif method == 'Levenberg-Marquardt':
        return _minimize_levenberg_marquardt(model, x0, cl_runtime_info, options)
    elif method == 'Subplex':
        return _minimize_subplex(model, x0, cl_runtime_info, options)
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
                'patience_line_search': 5,
                'reset_method': 'EXTRAPOLATED_POINT'}

    elif method == 'Nelder-Mead':
        return {'patience': 200,
                'alpha': 1.0, 'beta': 0.5, 'gamma': 2.0, 'delta': 0.5, 'scale': 1.0,
                'adaptive_scales': True}

    elif method == 'Levenberg-Marquardt':
        return {'patience': 250,
                'step_bound': 100.0, 'scale_diag': 1, 'usertol_mult': 30}

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


def _minimize_powell(model, x0, cl_runtime_info, options=None):
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

    all_kernel_data = dict(model.get_kernel_data())
    all_kernel_data.update({
        '_parameters': Array(x0, ctype='mot_float_type', is_writable=True, is_readable=True),
        '_return_codes': Zeros((nmr_problems,), ctype='char', is_readable=False, is_writable=True),
        '_tmp_likelihoods': LocalMemory('double')
    })

    objective_function = model.get_objective_function()
    cl_extra = objective_function.get_cl_code()
    cl_extra += '''
        double evaluate(local mot_float_type* x, void* data_void){
            mot_data_struct* data = (mot_data_struct*)data_void;
            return ''' + objective_function.get_cl_function_name() + '''(data, x, 0, data->_tmp_likelihoods);
        }
    '''

    params = {'NMR_PARAMS': nmr_parameters}
    for option, value in options.items():
        params.update({option.upper(): value})
    params['RESET_METHOD'] = 'POWELL_RESET_METHOD_' + params['RESET_METHOD'].upper()
    cl_extra += (open(os.path.abspath(resource_filename('mot', 'data/opencl/powell.cl')), 'r').read() % params)

    optimizer_func = SimpleCLFunction.from_string('''
        void compute(mot_data_struct* data){
            local mot_float_type x[''' + str(nmr_parameters) + '''];

            if(get_local_id(0) == 0){
                for(uint i = 0; i < ''' + str(nmr_parameters) + '''; i++){
                    x[i] = data->_parameters[i];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            char return_code = (char) powell(x, (void*)data);

            if(get_local_id(0) == 0){
                *data->_return_codes = return_code;

                for(uint i = 0; i < ''' + str(nmr_parameters) + '''; i++){
                    data->_parameters[i] = x[i];
                }   
            }
        }
    ''', cl_extra=cl_extra)
    
    optimizer_func.evaluate({'data': all_kernel_data}, nmr_instances=nmr_problems,
                            use_local_reduction=all(env.is_gpu for env in cl_runtime_info.cl_environments),
                            cl_runtime_info=cl_runtime_info)

    return OptimizeResults({'x': all_kernel_data['_parameters'].get_data(),
                            'status': all_kernel_data['_return_codes'].get_data()})


def _minimize_nmsimplex(model, x0, cl_runtime_info, options=None):
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

    all_kernel_data = dict(model.get_kernel_data())
    all_kernel_data.update({
        '_parameters': Array(x0, ctype='mot_float_type', is_writable=True, is_readable=True),
        '_return_codes': Zeros((nmr_problems,), ctype='char', is_readable=False, is_writable=True),
        '_tmp_likelihoods': LocalMemory('double')
    })

    objective_function = model.get_objective_function()
    cl_extra = objective_function.get_cl_code()
    cl_extra += '''
            double evaluate(local mot_float_type* x, void* data_void){
                mot_data_struct* data = (mot_data_struct*)data_void;
                return ''' + objective_function.get_cl_function_name() + '''(data, x, 0, data->_tmp_likelihoods);
            }
        '''

    params = {'NMR_PARAMS': nmr_parameters}
    for option, value in options.items():
        if option == 'scale':
            s = ''
            for ind in range(nmr_parameters):
                s += 'initial_simplex_scale[{}] = {};'.format(ind, value)
            params['INITIAL_SIMPLEX_SCALES'] = s
        else:
            params.update({option.upper(): value})

    if options['adaptive_scales']:
        params.update(
            {'ALPHA': 1,
             'BETA': 0.75 - 1.0 / (2 * nmr_parameters),
             'GAMMA': 1 + 2.0 / nmr_parameters,
             'DELTA': 1 - 1.0 / nmr_parameters}
        )

    lib_nmsimplex = LibNMSimplex(evaluate_fname='evaluate')
    cl_extra += (lib_nmsimplex.get_cl_code() + '''
        int nmsimplex(local mot_float_type* const model_parameters, void* data){
            local mot_float_type initial_simplex_scale[%(NMR_PARAMS)r];
            %(INITIAL_SIMPLEX_SCALES)s
            
            mot_float_type fdiff;
            mot_float_type psi = 0;
            local mot_float_type nmsimplex_scratch[
                %(NMR_PARAMS)r * 2 + (%(NMR_PARAMS)r + 1) * (%(NMR_PARAMS)r + 1)];

            return lib_nmsimplex(%(NMR_PARAMS)r, model_parameters, data, initial_simplex_scale,
                                 &fdiff, psi, (int)(%(PATIENCE)r * (%(NMR_PARAMS)r+1)),
                                 %(ALPHA)r, %(BETA)r, %(GAMMA)r, %(DELTA)r,
                                 nmsimplex_scratch);
        }
    ''' % params)

    optimizer_func = SimpleCLFunction.from_string('''
        void compute(mot_data_struct* data){
            local mot_float_type x[''' + str(nmr_parameters) + '''];

            if(get_local_id(0) == 0){
                for(uint i = 0; i < ''' + str(nmr_parameters) + '''; i++){
                    x[i] = data->_parameters[i];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            char return_code = (char) nmsimplex(x, (void*)data);

            if(get_local_id(0) == 0){
                *data->_return_codes = return_code;

                for(uint i = 0; i < ''' + str(nmr_parameters) + '''; i++){
                    data->_parameters[i] = x[i];
                }   
            }
        }
    ''', cl_extra=cl_extra)

    optimizer_func.evaluate({'data': all_kernel_data}, nmr_instances=nmr_problems,
                            use_local_reduction=all(env.is_gpu for env in cl_runtime_info.cl_environments),
                            cl_runtime_info=cl_runtime_info)

    return OptimizeResults({'x': all_kernel_data['_parameters'].get_data(),
                            'status': all_kernel_data['_return_codes'].get_data()})


def _minimize_subplex(model, x0, cl_runtime_info, options=None):
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

    all_kernel_data = dict(model.get_kernel_data())
    all_kernel_data.update({
        '_parameters': Array(x0, ctype='mot_float_type', is_writable=True, is_readable=True),
        '_return_codes': Zeros((nmr_problems,), ctype='char', is_readable=False, is_writable=True),
        '_tmp_likelihoods': LocalMemory('double')
    })

    objective_function = model.get_objective_function()
    cl_extra = objective_function.get_cl_code()
    cl_extra += '''
        double evaluate(local mot_float_type* x, void* data_void){
            mot_data_struct* data = (mot_data_struct*)data_void;
            return ''' + objective_function.get_cl_function_name() + '''(data, x, 0, data->_tmp_likelihoods);
        }
    '''

    params = {'NMR_PARAMS': nmr_parameters}
    for option, value in options.items():
        if option == 'scale':
            s = ''
            for ind in range(nmr_parameters):
                s += 'initial_simplex_scale[{}] = {};'.format(ind, value)
            params['INITIAL_SIMPLEX_SCALES'] = s
        elif option == 'adaptive_scales':
            params['ADAPTIVE_SCALES'] = int(bool(value))
        else:
            params.update({option.upper(): value})

    if options['min_subspace_length'] == 'auto':
        params.update({'MIN_SUBSPACE_LENGTH': min(2, nmr_parameters)})

    if options['max_subspace_length'] == 'auto':
        params.update({'MAX_SUBSPACE_LENGTH': min(5, nmr_parameters)})

    cl_extra += (open(os.path.abspath(resource_filename('mot', 'data/opencl/subplex.cl')), 'r').read() % params)
    cl_extra += LibNMSimplex(evaluate_fname='subspace_evaluate').get_cl_code()

    optimizer_func = SimpleCLFunction.from_string('''
        void compute(mot_data_struct* data){
            local mot_float_type x[''' + str(nmr_parameters) + '''];

            if(get_local_id(0) == 0){
                for(uint i = 0; i < ''' + str(nmr_parameters) + '''; i++){
                    x[i] = data->_parameters[i];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            char return_code = (char) subplex(x, (void*)data);

            if(get_local_id(0) == 0){
                *data->_return_codes = return_code;

                for(uint i = 0; i < ''' + str(nmr_parameters) + '''; i++){
                    data->_parameters[i] = x[i];
                }   
            }
        }
    ''', cl_extra=cl_extra)

    optimizer_func.evaluate({'data': all_kernel_data}, nmr_instances=nmr_problems,
                            use_local_reduction=all(env.is_gpu for env in cl_runtime_info.cl_environments),
                            cl_runtime_info=cl_runtime_info)

    return OptimizeResults({'x': all_kernel_data['_parameters'].get_data(),
                            'status': all_kernel_data['_return_codes'].get_data()})


def _minimize_levenberg_marquardt(model, x0, cl_runtime_info, options=None):
    options = _clean_options('Levenberg-Marquardt', options)

    nmr_problems = x0.shape[0]
    nmr_parameters = x0.shape[1]

    if model.get_nmr_observations() < x0.shape[1]:
        raise ValueError('The number of instances per problem must be greater than the number of parameters')

    all_kernel_data = dict(model.get_kernel_data())
    all_kernel_data.update({
        '_parameters': Array(x0, ctype='mot_float_type', is_writable=True, is_readable=True),
        '_return_codes': Zeros((nmr_problems,), ctype='char', is_readable=False, is_writable=True),
        '_tmp_likelihoods': LocalMemory('double'),
        '_fjac_all': Zeros((nmr_problems,
                            nmr_parameters,
                            model.get_nmr_observations()), ctype='mot_float_type',
                           is_writable=True, is_readable=True)
    })

    objective_function = model.get_objective_function()
    cl_extra = objective_function.get_cl_code()
    cl_extra += '''
        void evaluate(local mot_float_type* x, void* data_void, local mot_float_type* result){
            mot_data_struct* data = (mot_data_struct*)data_void;
            
            ''' + objective_function.get_cl_function_name() + '''(data, x, result, data->_tmp_likelihoods);
            
            // The LM method automatically squares the results, but the model also already does this.
            if(get_local_id(0) == 0){
                for(uint i = 0; i < ''' + str(model.get_nmr_observations()) + '''; i++){
                    result[i] = sqrt(fabs(result[i]));
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }    
    '''

    params = {'NMR_PARAMS': nmr_parameters, 'NMR_OBSERVATIONS': model.get_nmr_observations()}
    for option, value in options.items():
        if option == 'scale_diag':
            params['SCALE_DIAG'] = int(bool(value))
        else:
            params.update({option.upper(): value})
    cl_extra += (open(os.path.abspath(resource_filename('mot', 'data/opencl/lmmin.cl')), 'r').read() % params)

    optimizer_func = SimpleCLFunction.from_string('''
        void compute(mot_data_struct* data){
            local mot_float_type x[''' + str(nmr_parameters) + '''];

            if(get_local_id(0) == 0){
                for(uint i = 0; i < ''' + str(nmr_parameters) + '''; i++){
                    x[i] = data->_parameters[i];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            char return_code = (char) lmmin(x, (void*)data, data->_fjac_all);

            if(get_local_id(0) == 0){
                *data->_return_codes = return_code;

                for(uint i = 0; i < ''' + str(nmr_parameters) + '''; i++){
                    data->_parameters[i] = x[i];
                }   
            }
        }
        ''', cl_extra=cl_extra)

    optimizer_func.evaluate({'data': all_kernel_data}, nmr_instances=nmr_problems,
                            use_local_reduction=all(env.is_gpu for env in cl_runtime_info.cl_environments),
                            cl_runtime_info=cl_runtime_info)

    return OptimizeResults({'x': all_kernel_data['_parameters'].get_data(),
                            'status': all_kernel_data['_return_codes'].get_data()})
