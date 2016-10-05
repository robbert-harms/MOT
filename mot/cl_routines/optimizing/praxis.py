import os

from pkg_resources import resource_filename
from .base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class PrAxis(AbstractParallelOptimizer):

    default_patience = 1000

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=True, patience=None,
                 optimizer_options=None, **kwargs):
        """Use the Principal Axis method to calculate the optimum.

        This uses the Principal Axis implementation from NLOpt, slightly adapted for use in MOT.

        Possible optimizer options:

        - tolerance (float): default 0.0, praxis attempts to return praxis=f(x) such that if x0 is the
            true local minimum near x, then norm(x-x0) < t0 + squareroot(machep)*norm(x)
        - max_step_size (float):  default 1: Originally parameter H0. It is the maximum step size and
            should be set to about the max. distance from the initial guess to the minimum.
            If set too small or too large, the initial rate of convergence may be slow.
        - ill_conditioned (bool): if the problem is known to be ill-conditioned set it to 1 else, set to 0.
        - scbd (float): if the axes may be badly scaled (which is to be avoided if possible), then set SCBD=10.
            otherwise set SCBD=1.
        - ktm (int): KTM is the number of iterations without improvement before the algorithm terminates.
            KTM=4 is very cautious; usually KTM=1 is satisfactory.

        Args:
            patience (int):
                Used to set the maximum number of iterations to patience*(number_of_parameters+1)
            optimizer_options (dict): the optimization settings.
        """
        patience = patience or self.default_patience
        super(PrAxis, self).__init__(cl_environments, load_balancer, use_param_codec, patience=patience,
                                     optimizer_options=optimizer_options, **kwargs)

    def _get_worker_generator(self, *args):
        return lambda cl_environment: PrAxisWorker(cl_environment, *args)


class PrAxisWorker(AbstractParallelOptimizerWorker):

    def _get_optimization_function(self):
        params = {'NMR_PARAMS': self._nmr_params, 'PATIENCE': self._nmr_params}

        optimizer_options = self._optimizer_options or {}
        option_defaults = {'tolerance': 0.0, 'max_step_size': 1, 'ill_conditioned': False, 'scbd': 10., 'ktm': 1}
        option_converters = {'ill_conditioned': lambda val: int(bool(val))}

        for option, default in option_defaults.items():
            v = optimizer_options.get(option, default)
            if option in option_converters:
                v = option_converters[option](v)
            params.update({option.upper(): v})

        body = open(os.path.abspath(resource_filename('mot', 'data/opencl/praxis.pcl')), 'r').read()
        if params:
            body = body % params
        return body

    def _get_optimizer_call_name(self):
        return 'praxis'

    def _uses_random_numbers(self):
        return True
