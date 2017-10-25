import numpy as np
from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import KernelInputArray, KernelInputScalar, SimpleNamedCLFunction, KernelInputLocalMemory, \
    KernelInputAllocatedOutput
from ...cl_routines.base import CLRoutine


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class LogLikelihoodCalculator(CLRoutine):

    def calculate(self, model, parameters):
        """Calculate and return the log likelihood of the given model under the given parameters.

        This calculates log likelihoods for every problem in the model (typically after optimization),
        or a log likelihood for every sample of every model (typical after sampling). In the case of the first you
        can provide this function with a dictionary of parameters, or with an (d, p) array with d problems and
        p parameters. In the case of the second (after sampling), you must provide this function with a matrix of shape
        (d, p, n) with d problems, p parameters and n samples.

        Args:
            model (AbstractModel): The model to calculate the full log likelihood for.
            parameters (ndarray): The parameters to use in the evaluation of the model. This is either an (d, p) matrix
                or (d, p, n) matrix with d problems, p parameters and n samples.

        Returns:
            ndarray: per problem the log likelihood, or, per problem per sample the calculate log likelihood.
        """
        all_kernel_data = dict(model.get_kernel_data())
        all_kernel_data.update({
            'parameters': KernelInputArray(parameters),
        })

        shape = parameters.shape
        if len(shape) > 2:
            all_kernel_data.update({
                'log_likelihoods': KernelInputAllocatedOutput((shape[0], shape[2]), 'mot_float_type',
                                                              is_readable=False),
                'nmr_params': KernelInputScalar(parameters.shape[1]),
                'nmr_samples': KernelInputScalar(parameters.shape[2]),
                'local_reduction_lls': KernelInputLocalMemory('double')
            })
        else:
            all_kernel_data.update({
                'log_likelihoods': KernelInputAllocatedOutput((shape[0],), 'mot_float_type', is_readable=False),
                'local_reduction_lls': KernelInputLocalMemory('double')
            })

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._get_wrapped_function(model, parameters), all_kernel_data, parameters.shape[0],
                             double_precision=model.double_precision, use_local_reduction=True)

        return all_kernel_data['log_likelihoods'].get_data()

    def _get_wrapped_function(self, model, parameters):
        ll_func = model.get_log_likelihood_per_observation_function()
        nmr_params = parameters.shape[1]

        func = ''
        func += ll_func.get_cl_code()
        func += '''
            void _fill_log_likelihood_tmp(mot_data_struct* data,
                                          mot_float_type* x,
                                          local double* log_likelihood_tmp){

                ulong observation_ind;
                ulong local_id = get_local_id(0);
                log_likelihood_tmp[local_id] = 0;
                uint workgroup_size = get_local_size(0);
                uint elements_for_workitem = ceil(''' + str(model.get_nmr_inst_per_problem()) + ''' 
                                                  / (mot_float_type)workgroup_size);
                
                if(workgroup_size * (elements_for_workitem - 1) + local_id 
                        >= ''' + str(model.get_nmr_inst_per_problem()) + '''){
                    elements_for_workitem -= 1;
                }
                
                for(uint i = 0; i < elements_for_workitem; i++){
                    observation_ind = i * workgroup_size + local_id;
                
                    log_likelihood_tmp[local_id] += ''' + ll_func.get_cl_function_name() + '''(
                        data, x, observation_ind);
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            double _sum_log_likelihood_tmp(local double* log_likelihood_tmp){
                double ll = 0;
                for(uint i = 0; i < get_local_size(0); i++){
                    ll += log_likelihood_tmp[i];
                }
                return ll;
            }
            
        '''

        if len(parameters.shape) > 2:
            func += '''
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
            '''
        else:
            func += '''
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
            '''
        return SimpleNamedCLFunction(func, 'compute')
