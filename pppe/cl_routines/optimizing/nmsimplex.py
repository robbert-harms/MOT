from ...cl_functions import NMSimplexFunc
from ...cl_routines.optimizing.base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NMSimplex(AbstractParallelOptimizer):

    patience = 125

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=True, patience=patience):
        """Use the Nelder-Mead simplex method to calculate the optimimum.

        Args:
            patience (int):
                Used to set the maximum number of iterations to patience*(number_of_parameters+1)
        """
        super(NMSimplex, self).__init__(cl_environments, load_balancer, use_param_codec, patience=patience)

    def _get_worker(self, cl_environment, model, starting_points, full_output):
        return NMSimplexWorker(self, cl_environment, model, starting_points, full_output)


class NMSimplexWorker(AbstractParallelOptimizerWorker):

    def _get_optimization_function(self):
        return NMSimplexFunc(self._nmr_params, patience=self._parent_optimizer.patience)

    def _get_optimizer_call_name(self):
        return 'nmsimplex'