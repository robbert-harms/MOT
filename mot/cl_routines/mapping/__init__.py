from mot.cl_function import SimpleCLFunction
from mot.cl_routines.base import RunProcedure
from mot.kernel_data import KernelArray, KernelAllocatedArray, KernelScalar, KernelLocalMemory
import numpy as np


__author__ = 'Robbert Harms'
__date__ = "2014-05-21"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def calculate_dependent_parameters(kernel_data, estimated_parameters_list,
                                   parameters_listing, dependent_parameter_names, cl_runtime_info=None):
    """Calculate the dependent parameters

    Some of the models may contain parameter dependencies. We would like to return the maps for these parameters
    as well as all the other maps. Since the dependencies are specified in CL, we have to recourse to CL to
    calculate these maps.

    This uses the calculated parameters in the results dictionary to run the parameters_listing in CL to obtain
    the maps for the dependent parameters.

    Args:
        kernel_data (dict[str: mot.utils.KernelData]): the list of additional data to load
        estimated_parameters_list (list of ndarray): The list with the one-dimensional
            ndarray of estimated parameters
        parameters_listing (str): The parameters listing in CL
        dependent_parameter_names (list of list of str): Per parameter we would like to obtain the CL name and the
            result map name. For example: (('Wball_w', 'Wball.w'),)
        cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the runtime information

    Returns:
        dict: A dictionary with the calculated maps for the dependent parameters.
    """
    def get_cl_function():
        parameter_write_out = ''
        for i, p in enumerate([el[0] for el in dependent_parameter_names]):
            parameter_write_out += 'data->_results[' + str(i) + '] = ' + p + ";\n"

        body = '''
            mot_float_type x[''' + str(len(estimated_parameters_list)) + '''];

            for(uint i = 0; i < ''' + str(len(estimated_parameters_list)) + '''; i++){
                x[i] = data->x[i];
            }
            ''' + parameters_listing + '''
            ''' + parameter_write_out + '''
        '''
        return SimpleCLFunction('void', 'transform', ['mot_data_struct* data'], body)

    all_kernel_data = dict(kernel_data)
    all_kernel_data['x'] = KernelArray(np.dstack(estimated_parameters_list)[0, ...], ctype='mot_float_type')
    all_kernel_data['_results'] = KernelAllocatedArray(
        (estimated_parameters_list[0].shape[0], len(dependent_parameter_names)), 'mot_float_type')

    runner = RunProcedure(cl_runtime_info)
    runner.run_procedure(get_cl_function(), all_kernel_data, estimated_parameters_list[0].shape[0])

    return all_kernel_data['_results'].get_data()


def compute_log_likelihood(model, parameters, cl_runtime_info=None):
    """Calculate and return the log likelihood of the given model for the given parameters.

    This calculates the log likelihoods for every problem in the model (typically after optimization),
    or a log likelihood for every sample of every model (typically after sampling). In the case of the first (after
    optimization), the parameters must be an (d, p) array for d problems and p parameters. In the case of the
    second (after sampling), you must provide this function with a matrix of shape (d, p, n) with d problems,
    p parameters and n samples.

    Args:
        model (AbstractModel): The model to calculate the full log likelihood for.
        parameters (ndarray): The parameters to use in the evaluation of the model. This is either an (d, p) matrix
            or (d, p, n) matrix with d problems, p parameters and n samples.
        cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the runtime information

    Returns:
        ndarray: per problem the log likelihood, or, per problem and per sample the log likelihood.
    """

    def get_cl_function():
        ll_func = model.get_log_likelihood_per_observation_function()
        nmr_params = parameters.shape[1]

        ll_tmp_func = SimpleCLFunction.from_string('''
            void _fill_log_likelihood_tmp(mot_data_struct* data,
                                          mot_float_type* x,
                                          local double* log_likelihood_tmp){

                ulong observation_ind;
                ulong local_id = get_local_id(0);
                log_likelihood_tmp[local_id] = 0;
                uint workgroup_size = get_local_size(0);
                uint elements_for_workitem = ceil(''' + str(model.get_nmr_observations()) + ''' 
                                                  / (mot_float_type)workgroup_size);

                if(workgroup_size * (elements_for_workitem - 1) + local_id 
                        >= ''' + str(model.get_nmr_observations()) + '''){
                    elements_for_workitem -= 1;
                }

                for(uint i = 0; i < elements_for_workitem; i++){
                    observation_ind = i * workgroup_size + local_id;

                    log_likelihood_tmp[local_id] += ''' + ll_func.get_cl_function_name() + '''(
                        data, x, observation_ind);
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }
        ''', dependencies=[ll_func])

        ll_sum_func = SimpleCLFunction.from_string('''
            double _sum_log_likelihood_tmp(local double* log_likelihood_tmp){
                double ll = 0;
                for(uint i = 0; i < get_local_size(0); i++){
                    ll += log_likelihood_tmp[i];
                }
                return ll;
            }
        ''')

        if len(parameters.shape) > 2:
            return SimpleCLFunction.from_string('''
                void compute(mot_data_struct* data){
                    mot_float_type x[''' + str(nmr_params) + '''];

                    for(uint sample_ind = 0; sample_ind < ''' + str(parameters.shape[2]) + '''; sample_ind++){
                        for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                            x[i] = data->parameters[i *''' + str(parameters.shape[2]) + ''' + sample_ind];
                        }

                        _fill_log_likelihood_tmp(data, x, data->local_reduction_lls);
                        if(get_local_id(0) == 0){
                            data->log_likelihoods[sample_ind] = _sum_log_likelihood_tmp(data->local_reduction_lls);
                        }
                    }
                }
            ''', dependencies=[ll_tmp_func, ll_sum_func])

        return SimpleCLFunction.from_string('''
            void compute(mot_data_struct* data){
                mot_float_type x[''' + str(nmr_params) + '''];
                for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                    x[i] = data->parameters[i];
                }

                _fill_log_likelihood_tmp(data, x, data->local_reduction_lls);
                if(get_local_id(0) == 0){
                    *(data->log_likelihoods) = _sum_log_likelihood_tmp(data->local_reduction_lls);
                }
            }
        ''', dependencies=[ll_tmp_func, ll_sum_func])

    all_kernel_data = dict(model.get_kernel_data())
    all_kernel_data['parameters'] = KernelArray(parameters)

    shape = parameters.shape
    if len(shape) > 2:
        all_kernel_data.update({
            'log_likelihoods': KernelAllocatedArray((shape[0], shape[2]), 'mot_float_type'),
            'nmr_params': KernelScalar(parameters.shape[1]),
            'nmr_samples': KernelScalar(parameters.shape[2]),
            'local_reduction_lls': KernelLocalMemory('double')
        })
    else:
        all_kernel_data.update({
            'log_likelihoods': KernelAllocatedArray((shape[0],), 'mot_float_type'),
            'local_reduction_lls': KernelLocalMemory('double')
        })

    runner = RunProcedure(cl_runtime_info)
    runner.run_procedure(get_cl_function(), all_kernel_data, parameters.shape[0], use_local_reduction=True)

    return all_kernel_data['log_likelihoods'].get_data()


def compute_objective_value(model, parameters, cl_runtime_info=None):
    """Calculate and return the objective function value of the given model for the given parameters.

    Args:
        model (AbstractModel): The model to calculate the objective function for
        parameters (ndarray): The parameters to use in the evaluation of the model, an (d, p) matrix
            with d problems and p parameters.
        cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the runtime information

    Returns:
        ndarray: vector matrix with per problem the objective function value
    """

    def get_cl_function():
        objective_func = model.get_objective_per_observation_function()
        nmr_params = parameters.shape[1]

        fill_objective_func = SimpleCLFunction.from_string('''
            void _fill_objective_value_tmp(mot_data_struct* data,
                                          mot_float_type* x,
                                          local double* objective_value_tmp){

                ulong observation_ind;
                ulong local_id = get_local_id(0);
                objective_value_tmp[local_id] = 0;
                uint workgroup_size = get_local_size(0);
                uint elements_for_workitem = ceil(''' + str(model.get_nmr_observations()) + ''' 
                                                  / (mot_float_type)workgroup_size);

                if(workgroup_size * (elements_for_workitem - 1) + local_id 
                        >= ''' + str(model.get_nmr_observations()) + '''){
                    elements_for_workitem -= 1;
                }

                for(uint i = 0; i < elements_for_workitem; i++){
                    observation_ind = i * workgroup_size + local_id;
                    objective_value_tmp[local_id] += ''' + objective_func.get_cl_function_name() + '''(
                        data, x, observation_ind);
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }
        ''', dependencies=[objective_func])

        sum_objective_func = SimpleCLFunction.from_string('''
            double _sum_objective_value_tmp(local double* objective_value_tmp){
                double ll = 0;
                for(uint i = 0; i < get_local_size(0); i++){
                    ll += objective_value_tmp[i];
                }
                return ll;
            }
        ''')

        return SimpleCLFunction.from_string('''
            void compute(mot_data_struct* data){
                mot_float_type x[''' + str(nmr_params) + '''];
                for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                    x[i] = data->parameters[i];
                }

                _fill_objective_value_tmp(data, x, data->local_reduction_lls);
                if(get_local_id(0) == 0){
                    *(data->objective_values) = _sum_objective_value_tmp(data->local_reduction_lls);
                }
            }
        ''', dependencies=[fill_objective_func, sum_objective_func])

    shape = parameters.shape
    all_kernel_data = dict(model.get_kernel_data())
    all_kernel_data.update({
        'parameters': KernelArray(parameters),
        'objective_values': KernelAllocatedArray((shape[0],), 'mot_float_type'),
        'local_reduction_lls': KernelLocalMemory('double')
    })

    runner = RunProcedure(cl_runtime_info)
    runner.run_procedure(get_cl_function(), all_kernel_data, parameters.shape[0],
                         use_local_reduction=True)

    return all_kernel_data['objective_values'].get_data()

