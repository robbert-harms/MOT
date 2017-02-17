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

    def sample(self, model, init_params=None, full_output=False):
        """Minimize the given model with the given codec using the given environments.

        Args:
            model (SampleModelInterface): the model to minimize
            init_params (dict): a dictionary containing the results of a previous run, provides the starting point
            full_output (boolean): If true, also return other output parameters. It will then return a tuple like
                (samples, other_output_maps, proposal_state). If False, we only return the samples.

        Returns:
            dict or tuple: if full output is False we return only the samples in a dictionary.
                If full output is true it returns a tuple with as first elements the samples dict, as second
                an volumetric map dictionary and as last element a dictionary with the adaptive proposal values.
        """
