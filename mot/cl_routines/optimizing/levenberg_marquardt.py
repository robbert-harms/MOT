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

    def _get_worker_generator(self, *args):
        return lambda cl_environment: LevenbergMarquardtWorker(cl_environment, *args)


class LevenbergMarquardtWorker(AbstractParallelOptimizerWorker):

    def __init__(self, *args, **kwargs):
        super(LevenbergMarquardtWorker, self).__init__(*args, **kwargs)

        if self._model.get_nmr_inst_per_problem() < self._nmr_params:
            raise ValueError('The number of instances per problem must be greater than the number of parameters')

    def _get_evaluate_function(self):
        """Get the CL code for the evaluation function. This is called from _get_optimizer_cl_code.

        Implementing optimizers can change this if desired.

        Returns:
            str: the evaluation function.
        """
        kernel_source = ''
        kernel_source += self._model.get_objective_list_function('calculateObjectiveList')
        if self._use_param_codec:
            kernel_source += '''
                void evaluate(mot_float_type* x, const void* data, mot_float_type* result){
                    mot_float_type x_model[''' + str(self._nmr_params) + '''];
                    for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        x_model[i] = x[i];
                    }
                    decodeParameters(x_model);
                    calculateObjectiveList((optimize_data*)data, x_model, result);
                }
            '''
        else:
            kernel_source += '''
                void evaluate(mot_float_type* x, const void* data, mot_float_type* result){
                    calculateObjectiveList((optimize_data*)data, x, result);
                }
            '''
        return kernel_source

    def _get_optimization_function(self):
        return LMMin(self._nmr_params, self._model.get_nmr_inst_per_problem(),
                     patience=self._parent_optimizer.patience,
                     optimizer_options=self._optimizer_options)

    def _get_optimizer_call_name(self):
        return 'lmmin'
