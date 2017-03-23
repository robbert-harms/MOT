import logging
from ...cl_routines.base import CLRoutine

__author__ = 'Robbert Harms'
__date__ = "2014-05-18"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractSampler(CLRoutine):

    def __init__(self, cl_environments=None, load_balancer=None, **kwargs):
        super(AbstractSampler, self).__init__(cl_environments=cl_environments, load_balancer=load_balancer, **kwargs)
        self._logger = logging.getLogger(__name__)

    def sample(self, model, init_params=None):
        """Sample the given model with the given codec using the given environments.

        Args:
            model (SampleModelInterface): the model to sample
            init_params (dict): a dictionary containing the results of a previous run, provides the starting point

        Returns:
            SamplingOutput: the sampling output object
        """
        raise NotImplementedError()


class SamplingOutput(object):

    def get_samples(self):
        """Get the matrix containing the sampling results.

        Returns:
            ndarray: the sampled parameter maps, an (d, p, n) array with for d problems and p parameters n samples.
        """
        raise NotImplementedError()


class SimpleSampleOutput(SamplingOutput):

    def __init__(self, samples):
        """Simple storage container for the sampling output"""
        self._samples = samples

    def get_samples(self):
        return self._samples
