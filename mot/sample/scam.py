import numpy as np

from mot.lib.utils import is_scalar
from mot.sample.base import AbstractRWMSampler
from mot.lib.kernel_data import Array

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SingleComponentAdaptiveMetropolis(AbstractRWMSampler):

    def __init__(self, ll_func, log_prior_func, x0, proposal_stds,
                 waiting_period=100, scaling_factor=2.4, epsilon=1e-5, **kwargs):
        r"""An implementation of the Single Component Adaptive Metropolis (SCAM) MCMC algorithm [1].

        The SCAM method works by adapting the proposal standard deviation to the empirical standard deviation of the
        component's marginal distribution. That is, the standard deviation :math:`\sigma_i^{(t)}` for the proposal
        distribution of the :math:`i` th component at time :math:`t` is given by:

        .. math::

            \sigma_i^{(t)} = \begin{cases}
            \sigma_i^{(0)}, & t \leq t_s \\
            2.4 * \sqrt{\mathrm{Var}(\mathbf{X}^{(0)}_i, \ldots, \mathbf{X}^{(t-1)}_i)} + 1\cdot \epsilon, & t > t_s
            \end{cases}


        where :math:`t_s` denotes the iteration after which the adaptation starts (we use :math:`t_s = 100`).
        A small constant is necessary to prevent the standard deviation from shrinking to zero.
        This adaptation algorithm has been proven to retain ergodicity, meaning it is guaranteed to converge to the
        right stationary distribution [1].

        Args:
            ll_func (mot.lib.cl_function.CLFunction): The log-likelihood function. See parent docs.
            log_prior_func (mot.lib.cl_function.CLFunction): The log-prior function. See parent docs.
            x0 (ndarray): the starting positions for the sampler. Should be a two dimensional matrix
                with for every modeling instance (first dimension) and every parameter (second dimension) a value.
            proposal_stds (ndarray): for every parameter and every modeling instance an initial proposal std.
            waiting_period (int): only start updating the proposal std. after this many draws.
            scaling_factor (float): the scaling factor to use (the parameter ``s`` in the paper referenced).
            epsilon (float or ndarray): small number to prevent the std. from collapsing to zero.
                Can either be one value for all parameters, or one value per parameter.

        References:
            [1] Haario, H., Saksman, E., & Tamminen, J. (2005). Componentwise adaptation for high dimensional MCMC.
                Computational Statistics, 20(2), 265-273. https://doi.org/10.1007/BF02789703
        """
        super().__init__(ll_func, log_prior_func, x0, proposal_stds, **kwargs)
        self._waiting_period = waiting_period
        self._scaling_factor = scaling_factor
        self._epsilon = epsilon

        if is_scalar(self._epsilon):
            self._epsilon = np.ones(x0.shape[1]) * self._epsilon

        self._parameter_means = np.zeros((self._nmr_problems, self._nmr_params),
                                         dtype=self._cl_runtime_info.mot_float_dtype, order='C')
        self._parameter_variances = np.zeros((self._nmr_problems, self._nmr_params),
                                             dtype=self._cl_runtime_info.mot_float_dtype, order='C')
        self._parameter_variance_update_m2s = np.zeros((self._nmr_problems, self._nmr_params),
                                                       dtype=self._cl_runtime_info.mot_float_dtype, order='C')

    def _get_mcmc_method_kernel_data_elements(self):
        kernel_data = super()._get_mcmc_method_kernel_data_elements()
        kernel_data.update({
            'parameter_means': Array(self._parameter_means, 'mot_float_type', mode='rw', ensure_zero_copy=True),
            'parameter_variances': Array(self._parameter_variances, 'mot_float_type', mode='rw',
                                         ensure_zero_copy=True),
            'parameter_variance_update_m2s': Array(self._parameter_variance_update_m2s, 'mot_float_type',
                                                   mode='rw', ensure_zero_copy=True),
            'epsilons': Array(self._epsilon, 'float', mode='r', offset_str='0')
        })
        return kernel_data

    def _get_proposal_update_function(self, nmr_samples, thinning, return_output):
        kernel_source = '''
            /** Online variance algorithm by Welford:
             *      B. P. Welford (1962)."Note on a method for calculating corrected sums of squares
             *      and products". Technometrics 4(3):419-420.
             *
             * Also studied in:
             *      Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1983).
             *      Algorithms for Computing the Sample Variance: Analysis and Recommendations.
             *      The American Statistician 37, 242-247. http://www.jstor.org/stable/2683386
             */
            void _update_chain_statistics(ulong current_iteration,
                                          const mot_float_type new_param_value,
                                          global mot_float_type* const parameter_mean,
                                          global mot_float_type* const parameter_variance,
                                          global mot_float_type* const parameter_variance_update_m2){

                mot_float_type previous_mean = *parameter_mean;
                *parameter_mean += (new_param_value - *parameter_mean) / (current_iteration + 1);
                *parameter_variance_update_m2 += (new_param_value - previous_mean)
                                                    * (new_param_value - *parameter_mean);

                if(current_iteration > 1){
                    *parameter_variance = *parameter_variance_update_m2 / (current_iteration - 1);
                }
            }
            
            void _updateProposalState(_mcmc_method_data* method_data, ulong current_iteration, 
                                      global mot_float_type* current_position){    
                for(uint k = 0; k < ''' + str(self._nmr_params) + '''; k++){
                    _update_chain_statistics(current_iteration, current_position[k],
                                             method_data->parameter_means + k, 
                                             method_data->parameter_variances + k,
                                             method_data->parameter_variance_update_m2s + k);
                    
                    if(current_iteration > ''' + str(self._waiting_period) + '''){
                        method_data->proposal_stds[k] = ''' + str(self._scaling_factor) + ''' 
                            * (sqrt(method_data->parameter_variances[k]) + method_data->epsilons[k]);
                    }
                }                        
            }
        '''
        return kernel_source
