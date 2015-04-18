from ...tools import get_cl_double_extension_definer, ParameterCLCodeGenerator
from ...cl_functions import NMSimplexFunc
from ...cl_routines.optimizing.base import AbstractParallelOptimizer

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

    def _get_optimization_function(self, data_state):
        return NMSimplexFunc(data_state.nmr_params, patience=self.patience)

    def _get_optimizer_call_name(self):
        return 'nmsimplex'