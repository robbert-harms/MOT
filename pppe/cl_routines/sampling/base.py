from ...cl_environments import CLEnvironmentFactory
from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import PreferGPU

__author__ = 'Robbert Harms'
__date__ = "2014-05-18"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractSampler(AbstractCLRoutine):

    def __init__(self, cl_environments=None, load_balancer=None):
        """Sample the given model using the given environments and load balancers

        Args:
            cl_environments: a list with the cl environments to use
            load_balancer: the load balance strategy to use
        """
        if not load_balancer:
            load_balancer = PreferGPU()

        if not cl_environments:
            cl_environments = CLEnvironmentFactory.all_devices()

        super(AbstractSampler, self).__init__(cl_environments, load_balancer)

    def sample(self, model, init_params=None, full_output=False):
        """Minimize the given model with the given codec using the given environments.

        Args:
            model (SampleModelInterface): the model to minimize
            init_params (dict): a dictionary containing the results of a previous run, provides the starting point
            full_output (boolean): If true, also return other output parameters:
                (samples, {other_output}). Else only return the samples.
        """