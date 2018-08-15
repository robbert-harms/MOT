from mot.sample.base import AbstractRWMSampler

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MetropolisWithinGibbs(AbstractRWMSampler):

    def __init__(self, ll_func, log_prior_func, x0, proposal_stds, **kwargs):
        r"""An implementation of the Metropolis-Within-Gibbs MCMC algorithm [1].

        This does not scale the proposal standard deviations during sampling.

        Args:
            ll_func (mot.lib.cl_function.CLFunction): The log-likelihood function. See parent docs.
            log_prior_func (mot.lib.cl_function.CLFunction): The log-prior function. See parent docs.
            x0 (ndarray): the starting positions for the sampler. Should be a two dimensional matrix
                with for every modeling instance (first dimension) and every parameter (second dimension) a value.
            proposal_stds (ndarray): for every parameter and every modeling instance an initial proposal std.

        References:
            [1] 1. van Ravenzwaaij D, Cassey P, Brown SD. A simple introduction to Markov Chain Monte–Carlo sampling.
                Psychon Bull Rev. 2016:1-12. doi:10.3758/s13423-016-1015-8.
        """
        super().__init__(ll_func, log_prior_func, x0, proposal_stds, **kwargs)
