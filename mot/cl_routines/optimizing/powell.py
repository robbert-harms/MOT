import os
from pkg_resources import resource_filename
from .base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Powell(AbstractParallelOptimizer):

    default_patience = 5

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=True, patience=None,
                 optimizer_options=None, **kwargs):
        """Use the Powell method to calculate the optimum.

        Args:
            patience (int):
                Used to set the maximum number of iterations to patience*(number_of_parameters+1)
            optimizer_options (dict): the optimization settings, you can use the following:
                bracket_gold (double): the default ratio by which successive intervals are magnified in Bracketing
                glimit (double): the maximum magnification allowed for a parabolic-fit step in Bracketing
        """
        patience = patience or self.default_patience
        super(Powell, self).__init__(cl_environments, load_balancer, use_param_codec, patience=patience,
                                     optimizer_options=optimizer_options, **kwargs)

    def _get_worker_generator(self, *args):
        return lambda cl_environment: PowellWorker(cl_environment, *args)


class PowellWorker(AbstractParallelOptimizerWorker):

    def _get_optimization_function(self):
        params = {'NMR_PARAMS': self._nmr_params, 'PATIENCE': self._parent_optimizer.patience}

        optimizer_options = self._optimizer_options or {}
        option_defaults = {'bracket_gold': 1.618034, 'glimit': 100.0}

        for option, default in option_defaults.items():
            params.update({option.upper(): optimizer_options.get(option, default)})

        body = open(os.path.abspath(resource_filename('mot', 'data/opencl/powell.pcl')), 'r').read()
        if params:
            body = body % params
        return body

    def _get_optimizer_call_name(self):
        return 'powell'
