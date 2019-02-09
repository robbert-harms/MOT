from mot.lib.cl_function import SimpleCLFunction
from mot.lib.kernel_data import Array, Zeros
from mot.cl_routines.numerical_differentiation import estimate_hessian

__author__ = 'Robbert Harms'
__date__ = "2014-05-21"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def compute_log_likelihood(ll_func, parameters, data=None, cl_runtime_info=None):
    """Calculate and return the log likelihood of the given model for the given parameters.

    This calculates the log likelihoods for every problem in the model (typically after optimization),
    or a log likelihood for every sample of every model (typically after sample). In the case of the first (after
    optimization), the parameters must be an (d, p) array for d problems and p parameters. In the case of the
    second (after sample), you must provide this function with a matrix of shape (d, p, n) with d problems,
    p parameters and n samples.

    Args:
        ll_func (mot.lib.cl_function.CLFunction): The log-likelihood function. A CL function with the signature:

                .. code-block:: c

                        double <func_name>(local const mot_float_type* const x, void* data);

        parameters (ndarray): The parameters to use in the evaluation of the model. This is either an (d, p) matrix
            or (d, p, n) matrix with d problems, p parameters and n samples.
        data (mot.lib.kernel_data.KernelData): the user provided data for the ``void* data`` pointer.
        cl_runtime_info (mot.configuration.CLRuntimeInfo): the runtime information

    Returns:
        ndarray: per problem the log likelihood, or, per problem and per sample the log likelihood.
    """

    def get_cl_function():
        nmr_params = parameters.shape[1]

        if len(parameters.shape) > 2:
            return SimpleCLFunction.from_string('''
                void compute(global mot_float_type* parameters, 
                             global mot_float_type* log_likelihoods,
                             void* data){
                             
                    local mot_float_type x[''' + str(nmr_params) + '''];

                    for(uint sample_ind = 0; sample_ind < ''' + str(parameters.shape[2]) + '''; sample_ind++){
                        for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                            x[i] = parameters[i *''' + str(parameters.shape[2]) + ''' + sample_ind];
                        }
                        
                        double ll = ''' + ll_func.get_cl_function_name() + '''(x, data);
                        if(get_local_id(0) == 0){
                            log_likelihoods[sample_ind] = ll;
                        }
                    }
                }
            ''', dependencies=[ll_func])

        return SimpleCLFunction.from_string('''
            void compute(local mot_float_type* parameters, 
                         global mot_float_type* log_likelihoods,
                         void* data){
                         
                double ll = ''' + ll_func.get_cl_function_name() + '''(parameters, data);
                if(get_local_id(0) == 0){
                    *(log_likelihoods) = ll;
                }
            }
        ''', dependencies=[ll_func])

    kernel_data = {'data': data,
                   'parameters': Array(parameters, 'mot_float_type', mode='r')}

    shape = parameters.shape
    if len(shape) > 2:
        kernel_data.update({
            'log_likelihoods': Zeros((shape[0], shape[2]), 'mot_float_type'),
        })
    else:
        kernel_data.update({
            'log_likelihoods': Zeros((shape[0],), 'mot_float_type'),
        })

    get_cl_function().evaluate(kernel_data, parameters.shape[0], use_local_reduction=True,
                               cl_runtime_info=cl_runtime_info)

    return kernel_data['log_likelihoods'].get_data()


def compute_objective_value(objective_func, parameters, data=None, cl_runtime_info=None):
    """Calculate and return the objective function value of the given model for the given parameters.

    Args:
        objective_func (mot.lib.cl_function.CLFunction): A CL function with the signature:

            .. code-block:: c

                double <func_name>(local const mot_float_type* const x,
                                   void* data,
                                   local mot_float_type* objective_list);

        parameters (ndarray): The parameters to use in the evaluation of the model, an (d, p) matrix
            with d problems and p parameters.
        data (mot.lib.kernel_data.KernelData): the user provided data for the ``void* data`` pointer.
        cl_runtime_info (mot.configuration.CLRuntimeInfo): the runtime information

    Returns:
        ndarray: vector matrix with per problem the objective function value
    """
    return objective_func.evaluate({'data': data, 'parameters': Array(parameters, 'mot_float_type', mode='r')},
                                   parameters.shape[0], use_local_reduction=True, cl_runtime_info=cl_runtime_info)
