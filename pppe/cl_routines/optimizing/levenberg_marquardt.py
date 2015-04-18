from pppe.cl_functions import LMMin
from ...tools import get_cl_double_extension_definer, ParameterCLCodeGenerator
from .base import AbstractParallelOptimizer

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class LevenbergMarquardt(AbstractParallelOptimizer):

    patience = 250

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=True, patience=patience):
        """Use the Levenberg-Marquardt method to calculate the optimimum.

        Args:
            patience (int):
                Used to set the maximum number of iterations to patience*(number_of_parameters+1)
        """
        super(LevenbergMarquardt, self).__init__(cl_environments, load_balancer, use_param_codec)
        self.patience = patience

    def _get_optimizer_cl_code(self, data_state):
        param_codec = data_state.model.get_parameter_codec()
        use_param_codec = self.use_param_codec and param_codec and self._automatic_apply_codec

        optimizer_func = self._get_optimization_function(data_state)

        cl_eval_func = data_state.model.get_model_eval_function('evaluateModel')
        cl_observation_func = data_state.model.get_observation_return_function('getObservation')
        nmr_params = data_state.nmr_params
        param_codec = data_state.model.get_parameter_codec()
        nmr_inst_per_problem = data_state.model.get_nmr_inst_per_problem()

        if nmr_params <= 0:
            raise ValueError('The number of parameters can not be smaller or equal to 0.')

        if nmr_inst_per_problem < nmr_params:
            raise ValueError('The number of instances per problem must be greater than the number of parameters')

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + repr(nmr_inst_per_problem) + '''
        '''
        kernel_source += cl_observation_func
        kernel_source += cl_eval_func
        if use_param_codec:
            decode_func = param_codec.get_cl_decode_function('decodeParameters')
            kernel_source += decode_func + "\n"
            kernel_source += '''
                void evaluate(const void* data, double* x, double* result){
                    int i;
                    double model_x[''' + repr(nmr_params) + '''];
                    for(i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        model_x[i] = x[i];
                    }
                    decodeParameters(model_x);

                    for(i = 0; i < NMR_INST_PER_PROBLEM; i++){
                        result[i] = getObservation((optimize_data*)data, i) -
                                        evaluateModel((optimize_data*)data, model_x, i);
                    }
                }
            '''
        else:
            kernel_source += '''
                void evaluate(const void* data, double* x, double* result){
                    for(int i = 0; i < NMR_INST_PER_PROBLEM; i++){
                        result[i] = getObservation((optimize_data*)data, i) -
                                        evaluateModel((optimize_data*)data, x, i);
                    }
                }
            '''
        kernel_source += optimizer_func.get_cl_header()
        kernel_source += optimizer_func.get_cl_code()
        return kernel_source

    def _get_optimization_function(self, data_state):
        return LMMin(data_state.nmr_params, patience=self.patience)

    def _get_optimizer_call_name(self):
        return 'lmmin'
