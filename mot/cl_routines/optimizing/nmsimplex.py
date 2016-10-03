import os
from pkg_resources import resource_filename
from ...cl_routines.optimizing.base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NMSimplex(AbstractParallelOptimizer):

    default_patience = 200

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=True, patience=None,
                 optimizer_options=None, **kwargs):
        """Use the Nelder-Mead simplex method to calculate the optimimum.

        Args:
            patience (int):
                Used to set the maximum number of iterations to patience*(number_of_parameters+1)
            optimizer_options (dict): the optimization settings, you can use the following:
                - scale (double): the scale of the initial simplex
                - alpha (double): reflection coefficient
                - beta (double): contraction coefficient
                - gamma (double); expansion coefficient

                For the defaults please see NMSimplexFunc.
        """
        patience = patience or self.default_patience
        super(NMSimplex, self).__init__(cl_environments, load_balancer, use_param_codec, patience=patience,
                                        optimizer_options=optimizer_options, **kwargs)

    def _get_worker_generator(self, *args):
        return lambda cl_environment: NMSimplexWorker(cl_environment, *args)


class NMSimplexWorker(AbstractParallelOptimizerWorker):

    def _get_optimization_function(self):
        params = {'NMR_PARAMS': self._nmr_params, 'PATIENCE': self._parent_optimizer.patience}

        optimizer_options = self._optimizer_options or {}
        option_defaults = {'alpha': 1.0, 'beta': 0.5, 'gamma': 2.0, 'delta': 0.5, 'scale': 1.0}

        for option, default in option_defaults.items():
            if option == 'scale':
                params['INITIAL_SIMPLEX_SCALES'] = '{' + ', '.join([str(default)] * self._nmr_params) + '}'
            else:
                params.update({option.upper(): optimizer_options.get(option, default)})

        body = open(os.path.abspath(resource_filename('mot', 'data/opencl/nmsimplex.pcl')), 'r').read()
        if params:
            body = body % params
        return body

    def _get_optimizer_call_name(self):
        return 'nmsimplex'
