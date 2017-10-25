import numpy as np
from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import KernelInputArray, SimpleNamedCLFunction, KernelInputLocalMemory, KernelInputAllocatedOutput
from ...cl_routines.base import CLRoutine


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ObjectiveFunctionCalculator(CLRoutine):

    def calculate(self, model, parameters):
        """Calculate and return the objective function of the given model for the given parameters.

        Args:
            model (AbstractModel): The model to calculate the objective function for
            parameters (ndarray): The parameters to use in the evaluation of the model, an (d, p) matrix
                with d problems and p parameters.

        Returns:
            ndarray: per problem the objective function.
        """
        shape = parameters.shape
        all_kernel_data = dict(model.get_kernel_data())
        all_kernel_data.update({
            'parameters': KernelInputArray(parameters),
            'objective_values': KernelInputAllocatedOutput((shape[0],), 'mot_float_type', is_readable=False),
            'local_reduction_lls': KernelInputLocalMemory('double')
        })

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._get_wrapped_function(model, parameters), all_kernel_data, parameters.shape[0],
                             double_precision=model.double_precision, use_local_reduction=True)

        return all_kernel_data['objective_values'].get_data()

    def _get_wrapped_function(self, model, parameters):
        objective_func = model.get_objective_per_observation_function()
        nmr_params = parameters.shape[1]

        func = ''
        func += objective_func.get_cl_code()
        func += '''
            void _fill_objective_value_tmp(mot_data_struct* data,
                                          mot_float_type* x,
                                          local double* objective_value_tmp){

                ulong observation_ind;
                ulong local_id = get_local_id(0);
                objective_value_tmp[local_id] = 0;
                uint workgroup_size = get_local_size(0);
                uint elements_for_workitem = ceil(''' + str(model.get_nmr_inst_per_problem()) + ''' 
                                                  / (mot_float_type)workgroup_size);
                
                if(workgroup_size * (elements_for_workitem - 1) + local_id 
                        >= ''' + str(model.get_nmr_inst_per_problem()) + '''){
                    elements_for_workitem -= 1;
                }
                
                for(uint i = 0; i < elements_for_workitem; i++){
                    observation_ind = i * workgroup_size + local_id;
                    objective_value_tmp[local_id] += ''' + objective_func.get_cl_function_name() + '''(
                        data, x, observation_ind);
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            double _sum_objective_value_tmp(local double* objective_value_tmp){
                double ll = 0;
                for(uint i = 0; i < get_local_size(0); i++){
                    ll += objective_value_tmp[i];
                }
                return ll;
            }
            
        '''
        func += '''
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
        '''
        return SimpleNamedCLFunction(func, 'compute')
