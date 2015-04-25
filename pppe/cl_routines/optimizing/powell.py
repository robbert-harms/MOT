from pppe.cl_functions import PowellFunc
from ...utils import get_cl_double_extension_definer, ParameterCLCodeGenerator
from .base import AbstractParallelOptimizer

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Powell(AbstractParallelOptimizer):

    patience = 25

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=True, patience=patience):
        """Use the Powell method to calculate the optimimum.

        Args:
            patience (int):
                Used to set the maximum number of iterations to patience*(number_of_parameters+1)
        """
        super(Powell, self).__init__(cl_environments, load_balancer, use_param_codec, patience=patience)

    def _get_optimization_function(self, data_state):
        return PowellFunc(data_state.nmr_params, patience=self.patience)

    def _get_optimizer_call_name(self):
        return 'powell'