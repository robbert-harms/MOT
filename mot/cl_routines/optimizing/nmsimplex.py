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

    def __init__(self, patience=None, optimizer_settings=None, **kwargs):
        """Use the Nelder-Mead simplex method to calculate the optimimum.

        Args:
            patience (int):
                Used to set the maximum number of iterations to patience*(number_of_parameters+1)
            optimizer_settings (dict): the optimization settings, you can use the following:
                - scale (double): the scale of the initial simplex, default 1.0
                - alpha (double): reflection coefficient, default 1.0
                - beta (double): contraction coefficient, default 0.5
                - gamma (double); expansion coefficient, default 2.0
        """
        patience = patience or self.default_patience

        optimizer_settings = optimizer_settings or {}
        option_defaults = {'alpha': 1.0, 'beta': 0.5, 'gamma': 2.0, 'delta': 0.5, 'scale': 1.0}

        def get_value(option_name, default):
            value = optimizer_settings.get(option_name, None)
            if value is None:
                return default
            return value

        for option, default in option_defaults.items():
            optimizer_settings.update({option: get_value(option, default)})

        super(NMSimplex, self).__init__(patience=patience, optimizer_settings=optimizer_settings, **kwargs)

    def _get_worker_generator(self, *args):
        return lambda cl_environment: NMSimplexWorker(cl_environment, *args)


class NMSimplexWorker(AbstractParallelOptimizerWorker):

    def _get_optimization_function(self):
        params = {'NMR_PARAMS': self._nmr_params, 'PATIENCE': self._parent_optimizer.patience}

        for option, value in self._optimizer_settings.items():
            if option == 'scale':
                params['INITIAL_SIMPLEX_SCALES'] = '{' + ', '.join([str(value)] * self._nmr_params) + '}'
            else:
                params.update({option.upper(): value})

        body = open(os.path.abspath(resource_filename('mot', 'data/opencl/nmsimplex.pcl')), 'r').read()
        if params:
            body = body % params
        return body

    def _get_optimizer_call_name(self):
        return 'nmsimplex'
