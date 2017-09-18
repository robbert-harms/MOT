import numpy as np
from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import KernelInputBuffer, SimpleNamedCLFunction
from ...cl_routines.base import CLRoutine


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ResidualCalculator(CLRoutine):

    def calculate(self, model, parameters):
        """Calculate and return the residuals, that is the errors, per problem instance per data point.

        Args:
            model (AbstractModel): The model to calculate the residuals of.
            parameters (ndarray): The parameters to use in the evaluation of the model

        Returns:
            Return per voxel the errors (eval - data) per protocol item
        """
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        nmr_inst_per_problem = model.get_nmr_inst_per_problem()
        nmr_problems = model.get_nmr_problems()

        residuals = np.zeros((nmr_problems, nmr_inst_per_problem), dtype=np_dtype, order='C')

        all_kernel_data = dict(model.get_kernel_data())
        all_kernel_data.update({
            'parameters': KernelInputBuffer(parameters),
            'residuals': KernelInputBuffer(residuals, is_readable=False, is_writable=True)
        })

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._get_wrapped_function(model, parameters), all_kernel_data, parameters.shape[0],
                             double_precision=model.double_precision, use_local_reduction=False)

        return all_kernel_data['residuals'].get_data()

    def _get_wrapped_function(self, model, parameters):
        residual_function = model.get_residual_per_observation_function()
        param_modifier = model.get_pre_eval_parameter_modifier()
        nmr_params = parameters.shape[1]

        func = ''
        func += residual_function.get_cl_code()
        func += param_modifier.get_cl_code()
        func += '''
            void compute(mot_data_struct* data){
                mot_float_type x[''' + str(nmr_params) + '''];
                for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                    x[i] = data->parameters[i];
                }
                
                ''' + param_modifier.get_cl_function_name() + '''(data, x);
                
                for(uint i = 0; i < ''' + str(model.get_nmr_inst_per_problem()) + '''; i++){
                    data->residuals[i] = ''' + residual_function.get_cl_function_name() + '''(data, x, i);
                }
            }
        '''
        return SimpleNamedCLFunction(func, 'compute')
