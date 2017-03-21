import pyopencl as cl
import numpy as np
from ...utils import get_float_type_def
from ...cl_routines.base import CLRoutine
from ...load_balance_strategies import Worker


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class UnivariateEssAutoCorrelation(CLRoutine):
    """Calculate the univariate ESS for every chain and every problem instance.

    This estimates the effective sample size (ESS) using the autocorrelation of the chain.

    The ESS is an estimate of the size of an iid sample with the same variance as the current sample.
    This function implements the ESS as described in Kass et al. (1998) and Robert and Casella (2004; p. 500):

    .. math::

        ESS(X) = \frac{n}{\tau} = \frac{n}{1 + 2 * \sum_{k=1}^{m}{\rho_{k}}}

    where :math:`\rho_{k}` is estimated as:

    .. math::

        \hat{\rho}_{k} = \frac{E[(X_{t} - \mu)(X_{t + k} - \mu)]}{\sigma^{2}}

    References:
        * Kass, R. E., Carlin, B. P., Gelman, A., and Neal, R. (1998)
            Markov chain Monte Carlo in practice: A roundtable discussion. The American Statistician, 52, 93--100.
        * Robert, C. P. and Casella, G. (2004) Monte Carlo Statistical Methods. New York: Springer.
        * Geyer, C. J. (1992) Practical Markov chain Monte Carlo. Statistical Science, 7, 473--483.
    """

    def calculate(self, chains, max_lag=None):
        """Calculate and return the univariate ESS.

        The input can either be a 2d or a 3d matrix. If a 2d matrix of size (p, s) is given we assume it contains
        s samples for every p parameter of a single problem. If a 3d matrix of size (n, p, s) is given we assume it
        contains a matrix (p, s) for every n problems.

        This implementation calculates the averages and variances in double precision, regardless of the input dtype.

        Args:
            chains (ndarray): a 2d or 3d input matrix of size (p, s) or size (n, p, s) with n the number of problems,
                p the number of parameters and s the sample size
            max_lag (int): the maximum lag to use in the autocorrelation computation. If not given we use:
                :math:`min(n/3, 1000)`.

        Returns:
            ndarray: per problem instance and per parameter the ESS. If a 2d matrix is given we return a 1d array, if
                a 2d matrix is given we return a 2d matrix.
        """
        return_dtype = np.float32
        double_precision = False
        if chains.dtype == np.float64:
            return_dtype = np.float64
            double_precision = True

        return_2d = False
        if len(chains.shape) < 3:
            return_2d = True
            chains = chains[None, ...]

        max_lag = max_lag or min(chains.shape[2] // 3, 1000)

        ess_results = np.zeros(chains.shape[:2], dtype=return_dtype)

        workers = self._create_workers(lambda cl_environment: _UnivariateEssAutoCorrelationWorker(
            cl_environment, self.get_compile_flags_list(True), chains, ess_results, double_precision, max_lag))
        self.load_balancer.process(workers, chains.shape[0])

        if return_2d:
            return ess_results[0]
        return ess_results


class _UnivariateEssAutoCorrelationWorker(Worker):

    def __init__(self, cl_environment, compile_flags, chains, ess_results, double_precision, max_lag):
        super(_UnivariateEssAutoCorrelationWorker, self).__init__(cl_environment)

        self._chains = chains
        self._ess_results = ess_results
        self._double_precision = double_precision
        self._max_lag = max_lag

        self._nmr_problems = chains.shape[0]
        self._nmr_params = chains.shape[1]
        self._nmr_samples = chains.shape[2]

        self._all_buffers, self._results_buffer = self._create_buffers()
        self._kernel = self._build_kernel(compile_flags)

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start
        event = self._kernel.get_ess(self._cl_run_context.queue, (int(nmr_problems), int(self._nmr_params)),
                                     None, *self._all_buffers, global_offset=(int(range_start), 0))
        return [self._enqueue_readout(self._results_buffer, self._ess_results, range_start, range_end, [event])]

    def _create_buffers(self):
        chains_buffer = cl.Buffer(self._cl_run_context.context,
                                  cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                  hostbuf=self._chains)

        means_buffer = cl.Buffer(self._cl_run_context.context,
                                 cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                 hostbuf=np.mean(self._chains, dtype=np.float64, axis=2).astype(np.float32))

        errors_buffer = cl.Buffer(self._cl_run_context.context,
                                  cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                  hostbuf=self._ess_results)
        return [chains_buffer, means_buffer, errors_buffer], errors_buffer

    def _get_kernel_source(self):
        kernel_param_names = [
            'global mot_float_type chains[' + str(self._nmr_problems) + ']'
                                        '[' + str(self._nmr_params) + ']'
                                        '[' + str(self._nmr_samples) + ']',
            'global mot_float_type means[' + str(self._nmr_problems) + ']'
                                       '[' + str(self._nmr_params) + ']',
            'global mot_float_type ess_results[' + str(self._nmr_problems) + '][' + str(self._nmr_params) + ']']

        kernel_source = '''
            #define NMR_PROBLEMS ''' + str(self._nmr_problems) + '''
            #define NMR_PARAMS ''' + str(self._nmr_params) + '''
            #define NMR_SAMPLES ''' + str(self._nmr_samples) + '''
        '''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += '''
            /**
             * online variance, algorithm by Welford
             *  B. P. Welford (1962)."Note on a method for calculating corrected sums of squares
             *      and products". Technometrics 4(3):419–420.
             *
             * also studied in:
             * Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1983).
             *      Algorithms for Computing the Sample Variance: Analysis and Recommendations.
             *      The American Statistician 37, 242-247. http://www.jstor.org/stable/2683386
             *
             * This implementation returns the population variance.
             */
            void calculate_mean_and_variance(global mot_float_type* const array, const uint n,
                                             double* mean, double* variance){
                *mean = 0;
                double previous_mean;
                double m2 = 0;

                for(uint i = 0; i < n; i++){
                    previous_mean = *mean;
                    *mean += (array[i] - *mean) / (i + 1);
                    m2 += (array[i] - previous_mean) * (array[i] - *mean);
                }
                *variance = m2 / n;
            }

            /**
             * Calculate the mean of the normalized chain with the given lag (the auto-covariance).
             * In Python terms, this computes the following:
             *      normalized_chain = chain - mean(chain)
             *      mean(normalized_chain[:len(chain) - lag] * normalized_chain[lag:])
             */
            double get_auto_covariance(global mot_float_type* chain, const uint n, const uint lag,
                                       const double chain_mean){

                double sum = (chain[0] - chain_mean) * (chain[lag + 0] - chain_mean) / (n - lag);
                double c = 0.0;
                double y = 0.0;
                double t = 0.0;

                for(uint i = 1; i < (n - lag); i++){
                    y = ((chain[i] - chain_mean) * (chain[lag + i] - chain_mean) / (n - lag)) - c;
                    t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                return sum;
            }

            /**
             * Get the auto correlation time, here computed as the sum of autocorrelations with increasing lag.
             */
            double get_auto_correlation_time(global mot_float_type* chain, const uint n, mot_float_type mean){
                double mean2, variance;
                calculate_mean_and_variance(chain, n, &mean2, &variance);

                double previous_accoeff = variance;
                double auto_corr_sum = 0;
                double auto_correlation_coeff;

                for(uint lag = 1; lag < ''' + str(self._max_lag) + '''; lag++){
                    auto_correlation_coeff = get_auto_covariance(chain, n, lag, mean);

                    if(lag % 2 == 0){
                        if(previous_accoeff + auto_correlation_coeff <= 0){
                            break;
                        }
                    }

                    auto_corr_sum += auto_correlation_coeff;
                    previous_accoeff = auto_correlation_coeff;
                }

                return auto_corr_sum / variance;
            }

            __kernel void get_ess(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    uint problem_ind = get_global_id(0);
                    uint param_ind = get_global_id(1);

                    global mot_float_type* chain = chains[problem_ind][param_ind];
                    mot_float_type mean = means[problem_ind][param_ind];

                    ess_results[problem_ind][param_ind] =
                        (mot_float_type)(NMR_SAMPLES / (1 + 2 * get_auto_correlation_time(chain, NMR_SAMPLES, mean)));
            }
        '''
        return kernel_source
