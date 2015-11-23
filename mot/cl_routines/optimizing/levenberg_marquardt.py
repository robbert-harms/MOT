from ...cl_functions import LMMin
from .base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class LevenbergMarquardt(AbstractParallelOptimizer):

    default_patience = 250

    def __init__(self, cl_environments, load_balancer, use_param_codec=True, patience=None,
                 optimizer_options=None, **kwargs):
        """Use the Levenberg-Marquardt method to calculate the optimimum.

        Args:
            patience (int):
                Used to set the maximum number of iterations to patience*(number_of_parameters+1)
        """
        patience = patience or self.default_patience
        super(LevenbergMarquardt, self).__init__(cl_environments, load_balancer, use_param_codec, patience=patience,
                                                 optimizer_options=optimizer_options, **kwargs)

    def _get_worker_class(self):
        return LevenbergMarquardtWorker


class LevenbergMarquardtWorker(AbstractParallelOptimizerWorker):

    def _get_optimizer_cl_code(self):
        optimizer_func = self._get_optimization_function()

        cl_eval_func = self._model.get_model_eval_function('evaluateModel')
        cl_observation_func = self._model.get_observation_return_function('getObservation')
        nmr_params = self._nmr_params
        param_codec = self._model.get_parameter_codec()
        nmr_inst_per_problem = self._model.get_nmr_inst_per_problem()

        if nmr_params <= 0:
            raise ValueError('The number of parameters can not be smaller or equal to 0.')

        if nmr_inst_per_problem < nmr_params:
            raise ValueError('The number of instances per problem must be greater than the number of parameters')

        kernel_source = ''
        kernel_source += '''
            #define NMR_INST_PER_PROBLEM ''' + str(nmr_inst_per_problem) + '''
        '''
        kernel_source += cl_observation_func
        kernel_source += cl_eval_func

        if self._use_param_codec:
            decode_func = param_codec.get_cl_decode_function('decodeParameters')
            kernel_source += decode_func + "\n"

        kernel_source += '''
            void evaluate(const void* data, MOT_FLOAT_TYPE* x, MOT_FLOAT_TYPE* result){
                int i;
                MOT_FLOAT_TYPE x_model[''' + str(nmr_params) + '''];
                for(i = 0; i < ''' + str(nmr_params) + '''; i++){
                    x_model[i] = x[i];
                }

                ''' + ('decodeParameters(x_model);' if self._use_param_codec else '') + '''

                for(i = 0; i < NMR_INST_PER_PROBLEM; i++){
                    result[i] = getObservation((optimize_data*)data, i) -
                                    evaluateModel((optimize_data*)data, x_model, i);
                }
            }
        '''
        kernel_source += optimizer_func.get_cl_header()
        kernel_source += optimizer_func.get_cl_code()
        return kernel_source

    def _get_optimization_function(self):
        return LMMin(self._nmr_params, patience=self._parent_optimizer.patience,
                     optimizer_options=self._optimizer_options)

    def _get_optimizer_call_name(self):
        return 'lmmin'