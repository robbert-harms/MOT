from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import SimpleNamedCLFunction, KernelInputBuffer, KernelInputScalar
from ...cl_routines.base import CLRoutine
import numpy as np

__author__ = 'Robbert Harms'
__date__ = '2017-09-11'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class UnivariateESSStandardError(CLRoutine):

    def calculate(self, samples, batch_sizes=None, mcse_method='batch_means'):
        """Compute the univariate ESS using the standard error method.

        This computes the ESS using:

        .. math::

            ESS(X) = n * \frac{\lambda^{2}}{\sigma^{2}}

        Where :math:`\lambda` is the standard deviation of the chain and :math:`\sigma` is estimated using the
        monte carlo standard error (which in turn is, by default, estimated using a batch means estimator).

        For more information and an introduction to these methods, see the thesis of Flegal,
        "Monte Carlo Standard Errors for Markov Chain Monte Carlo", July 2008.

        Args:
            samples (ndarray): a matrix of shape (d, p, n) with d problems, p parameters and n samples
            batch_sizes (list or None): the size of the batches we will use for the moving average. If not given we
                a single batch using :math:`\floor(chain_length^{1/2})`
            mcse_method (str): the Monte Carlo Standard Error method to use for calculating the standard error of a
                chain. One of 'batch_means' or 'overlapping_batch_means'. Defaults to 'batch_means'.

        Returns:
            ndarray: a matrix of shape (d, p) with for every d problems and p parameters the univariate ESS of the
                samples

        References:
            * Flegal, J.M., Haran, M., and Jones, G.L. (2008). "Markov chain Monte Carlo: Can We
              Trust the Third Significant Figure?". Statistical Science, 23, p. 250–260.
            * Marc S. Meketon and Bruce Schmeiser. 1984. Overlapping batch means: something for nothing?.
              In Proceedings of the 16th conference on Winter simulation (WSC '84), Sallie Sheppard (Ed.).
              IEEE Press, Piscataway, NJ, USA, 226-230.
        """
        if mcse_method is None:
            mcse_method = 'batch_means'

        if mcse_method not in ('batch_means', 'overlapping_batch_means'):
            raise ValueError('The MCSE method must be None or one of "batch_means" or '
                             '"overlapping_batch_means", "{}" given.'.format(mcse_method))

        if batch_sizes is None:
            batch_sizes = [np.floor(np.sqrt(samples.shape[2]))]

        double_precision = samples.dtype == np.float64

        np_dtype = np.float32
        if double_precision:
            np_dtype = np.float64

        ess = np.zeros(samples.shape[0] * samples.shape[1], dtype=np_dtype, order='C')

        all_kernel_data = {'samples': KernelInputBuffer(np.reshape(samples, (-1, samples.shape[2]))),
                           'ess': KernelInputBuffer(ess, is_readable=False, is_writable=True),
                           'nmr_samples': KernelInputScalar(samples.shape[2]),
                           'batch_sizes': KernelInputBuffer(batch_sizes, offset_str='0'),
                           'nmr_batch_sizes': KernelInputScalar(len(batch_sizes))
                           }

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._get_wrapped_function(mcse_method), all_kernel_data,
                             samples.shape[0] * samples.shape[1],
                             double_precision=double_precision)

        return np.reshape(all_kernel_data['ess'].get_data(), (samples.shape[0], samples.shape[1]))

    def _get_wrapped_function(self, mcse_method):
        func = '''
            void _chain_stats(global mot_float_type* samples, uint nmr_samples, 
                              double* mean, double* variance){
                
                *variance = 0;
                *mean = 0;
                double delta;
                double value;
                
                for(uint i = 0; i < nmr_samples; i++){
                    value = samples[i];
                    delta = value - *mean;
                    *mean += delta / (i + 1);
                    *variance += delta * (value - *mean);
                }
                *variance /= nmr_samples;
            }
            
            void _chain_mean(global mot_float_type* samples, uint nmr_samples, double* mean){
                *mean = 0;
                double delta;
                double value;
                for(uint i = 0; i < nmr_samples; i++){
                    value = samples[i];
                    *mean += (value - *mean) / (i + 1);
                }
            }
            
            /**
             * Compute the Markov Chain Standard Error using batch means.
             * 
             * This will only compute the batch mean over complete batches and will ignore the last batch if
             * if does not have size equal to the batch size.
             * 
             * This returns the standard error of the mean :math:`\sigma / \sqrt{n}` where :math:`\sigma` is calculated
             * using batch means.
             */ 
            double _mcse_batch_means(mot_data_struct* data, uint batch_size, double chain_mean){
                uint nmr_batches = (uint)floor((double)data->nmr_samples / batch_size);
                
                double var_sum = 0;
                double batch_mean = 0;
                
                for(int i = 0; i < nmr_batches; i++){
                    _chain_mean(data->samples + i * batch_size, batch_size, &batch_mean);
                    var_sum += pown(batch_mean - chain_mean, 2);
                }
                
                double var_hat = (batch_size / (nmr_batches - 1)) * var_sum;
                return sqrt(var_hat / data->nmr_samples);                
            }
            
            /**
             * Computes the Monte Carlo Standard Error using overlapping batch means.
             *
             * This will only compute the batch mean over complete batches and will ignore the last batch if
             * if does not have size equal to the batch size.
             *
             * This returns the standard error of the mean :math:`\sigma / \sqrt{n}` where :math:`\sigma` is calculated
             * using overlapping batch means.
             */
            double _mcse_overlapping_batch_means(mot_data_struct* data, uint batch_size, double chain_mean){
                uint nmr_batches = (uint)(data->nmr_samples - batch_size + 1);
        
                double var_sum = 0;
                double batch_mean = 0;
                uint current_batch_size = batch_size;
                
                for(int i = 0; i < nmr_batches; i++){
                    if(i == nmr_batches - 1){
                        current_batch_size = min((uint)(data->nmr_samples - (nmr_batches - 1) * batch_size), 
                                                 (uint)batch_size);
                    }
                    
                    _chain_mean(data->samples + i, current_batch_size, &batch_mean);
                    var_sum += pown(batch_mean - chain_mean, 2);
                }
                
                double var_hat = ((data->nmr_samples * batch_size) / 
                    ((data->nmr_samples - batch_size + 1) * (data->nmr_samples - batch_size))) * var_sum;
                return sqrt(var_hat / data->nmr_samples);
            }
            
            double _monte_carlo_standard_error(mot_data_struct* data, double chain_mean){
                double min_mcse = INFINITY;
                double mcse;
                
                for(int i = 0; i < data->nmr_batch_sizes; i++){
                    mcse = _mcse_''' + mcse_method + '''(data, data->batch_sizes[i], chain_mean); 
                    
                    if(mcse < min_mcse){
                        min_mcse = mcse; 
                    }
                }
                return min_mcse;
            }
            
            void compute(mot_data_struct* data){
                double chain_mean;
                double chain_variance;
                _chain_stats(data->samples, data->nmr_samples, &chain_mean, &chain_variance);
                
                double sigma = pown(_monte_carlo_standard_error(data, chain_mean), 2) * data->nmr_samples;
                
                *(data->ess) = data->nmr_samples * (chain_variance / sigma);
            }            
        '''
        return SimpleNamedCLFunction(func, 'compute')
