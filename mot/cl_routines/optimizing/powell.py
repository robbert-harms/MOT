from ...cl_functions import PowellFunc
from .base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Powell(AbstractParallelOptimizer):

    default_patience = 5

    def __init__(self, cl_environments, load_balancer, use_param_codec=True, patience=None,
                 optimizer_options=None, **kwargs):
        """Use the Powell method to calculate the optimum.

        Args:
            patience (int):
                Used to set the maximum number of iterations to patience*(number_of_parameters+1)
            optimizer_options (dict): the optimization settings, you can use the following:
                - bracket_gold (double): the default ratio by which successive intervals are magnified in Bracketing
                - glimit (double): the maximum magnification allowed for a parabolic-fit step in Bracketing

                For the defaults please see PowellFunc.
        """
        patience = patience or self.default_patience
        super(Powell, self).__init__(cl_environments, load_balancer, use_param_codec, patience=patience,
                                     optimizer_options=optimizer_options, **kwargs)

    def _get_worker_class(self):
        return PowellWorker


class PowellWorker(AbstractParallelOptimizerWorker):

    def _get_optimization_function(self):
        return PowellFunc(self._nmr_params, patience=self._parent_optimizer.patience,
                          optimizer_options=self._optimizer_options)

    def _get_optimizer_call_name(self):
        return 'powell'

    def _optimizer_supports_float(self):
        return True

    def _optimizer_supports_double(self):
        return True