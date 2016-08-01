import logging
from mot.cl_routines.sampling.base import AbstractSampler
from mot.cl_routines.sampling.metropolis_hastings import MetropolisHastings

__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MetaSampler(AbstractSampler):

    def __init__(self, cl_environments=None, load_balancer=None, **kwargs):
        """Adds additional functionality to sampling.

        Args:
            cl_environments (list of CLEnvironment): a list with the cl environments to use
            load_balancer (LoadBalancer): the load balance strategy to use
            use_param_codec (boolean): if this minimization should use the parameter codecs (param transformations)
            patience (int): The patience is used in the calculation of how many iterations to iterate the sampler.
                The exact semantical value of this parameter may change per sampler.

        Attributes:
            sampler (sampler, default MetropolisHastings): The default sampling routine
        """
        super(MetaSampler, self).__init__(cl_environments, load_balancer, **kwargs)

        self.sampler = MetropolisHastings(self.cl_environments, self.load_balancer)

        self._propagate_property('cl_environments', cl_environments)
        self._propagate_property('load_balancer', load_balancer)

        self._logger = logging.getLogger(__name__)

    def sample(self, model, init_params=None, full_output=False):
        results = init_params
        results = self.sampler.sample(model, init_params=results, full_output=full_output)
        if full_output:
            return results[0], results[1]
        return results

    @property
    def cl_environments(self):
        return self._cl_environments

    @property
    def load_balancer(self):
        return self._load_balancer

    @cl_environments.setter
    def cl_environments(self, cl_environments):
        self._propagate_property('cl_environments', cl_environments)
        self._cl_environments = cl_environments

    @load_balancer.setter
    def load_balancer(self, load_balancer):
        self._propagate_property('load_balancer', load_balancer)
        self._load_balancer = load_balancer

    def _propagate_property(self, name, value):
        self.sampler.__setattr__(name, value)
