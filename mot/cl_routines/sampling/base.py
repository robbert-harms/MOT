import logging
from ...cl_routines.base import AbstractCLRoutine

__author__ = 'Robbert Harms'
__date__ = "2014-05-18"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractSampler(AbstractCLRoutine):

    def __init__(self, cl_environments, load_balancer):
        super(AbstractSampler, self).__init__(cl_environments, load_balancer)
        self._logger = logging.getLogger(__name__)

    def sample(self, model, init_params=None, full_output=False):
        """Minimize the given model with the given codec using the given environments.

        Args:
            model (SampleModelInterface): the model to minimize
            init_params (dict): a dictionary containing the results of a previous run, provides the starting point
            full_output (boolean): If true, also return other output parameters:
                (samples, {other_output}). Else only return the samples.
        """