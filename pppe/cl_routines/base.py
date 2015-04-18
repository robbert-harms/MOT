from ..cl_environments import CLEnvironmentFactory
from ..load_balance_strategies import PreferGPU

__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractCLRoutine(object):

    def __init__(self, cl_environments=None, load_balancer=None):
        """This class serves as an abstract basis for all CL routine classes.

        Args:
            cl_environments (list of CLEnvironment): The list of CL environments using by this routine.
            load_balancer (LoadBalancingStrategy): The load balancing strategy to be used by this routine.
        """
        if not load_balancer:
            load_balancer = PreferGPU()

        if not cl_environments:
            cl_environments = CLEnvironmentFactory.all_devices()

        self._cl_environments = cl_environments
        self._load_balancer = load_balancer

    @property
    def cl_environments(self):
        return self._cl_environments

    @cl_environments.setter
    def cl_environments(self, cl_environments):
        self._cl_environments = cl_environments

    @property
    def load_balancer(self):
        return self._load_balancer

    @load_balancer.setter
    def load_balancer(self, load_balancer):
        self._load_balancer = load_balancer