from ...cl_functions import PowellFunc
from .base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Powell(AbstractParallelOptimizer):

    default_patience = 5

    def __init__(self, cl_environments, load_balancer, use_param_codec=True, patience=None):
        """Use the Powell method to calculate the optimum.

        Args:
            patience (int):
                Used to set the maximum number of iterations to patience*(number_of_parameters+1)
        """
        patience = patience or self.default_patience
        super(Powell, self).__init__(cl_environments, load_balancer, use_param_codec, patience=patience)

    def _get_worker_class(self):
        return PowellWorker


class PowellWorker(AbstractParallelOptimizerWorker):

    def _get_optimization_function(self):
        return PowellFunc(self._nmr_params, patience=self._parent_optimizer.patience)

    def _get_optimizer_call_name(self):
        return 'powell'

    def _optimizer_supports_float(self):
        return True

    def _optimizer_supports_double(self):
        return True