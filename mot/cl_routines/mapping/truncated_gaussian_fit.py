from scipy.stats import truncnorm

from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import SimpleNamedCLFunction, KernelInputBuffer, KernelInputScalar
from ...cl_routines.base import CLRoutine
import numpy as np

__author__ = 'Robbert Harms'
__date__ = '2017-09-11'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class TruncatedGaussianFit(CLRoutine):

    def calculate(self, samples, high=np.pi, low=0):
        """Calculate the truncated normal mean and standard deviation.

        This uses the first two moments of the truncated normal distribution for approximating the fit.
        Please note that the sample means may lay outside of the high and low boundaries.

        This is basically a CL version of ``scipy.truncnorm.fit_loc_scale``.

        Args:
            samples (ndarray): the samples from which we want to calculate the mean and std.
            low (float): the low boundary for circular mean range
            high (float): the high boundary for circular mean range

        Returns:
            tuple: mean and std arrays
        """
        double_precision = samples.dtype == np.float64

        np_dtype = np.float32
        if double_precision:
            np_dtype = np.float64

        means = np.zeros(samples.shape[0], dtype=np_dtype, order='C')
        stds = np.zeros(samples.shape[0], dtype=np_dtype, order='C')
        samples = np.require(samples, np_dtype, requirements=['C', 'A', 'O'])

        mu, mu2 = truncnorm.stats(low, high, **{'moments': 'mv'})

        all_kernel_data = {'samples': KernelInputBuffer(samples),
                           'means': KernelInputBuffer(means, is_readable=False, is_writable=True),
                           'stds': KernelInputBuffer(stds, is_readable=False, is_writable=True),
                           'nmr_samples': KernelInputScalar(samples.shape[1]),
                           'mu': KernelInputScalar(mu),
                           'mu2': KernelInputScalar(mu2),
                           }

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._get_wrapped_function(), all_kernel_data, samples.shape[0],
                             double_precision=double_precision)

        return all_kernel_data['means'].get_data(), all_kernel_data['stds'].get_data()

    def _get_wrapped_function(self):
        func = '''
            void compute(mot_data_struct* data){
                double variance = 0;
                double mean = 0;
                double delta;
                double value;
                
                for(uint i = 0; i < data->nmr_samples; i++){
                    value = data->samples[i];
                    delta = value - mean;
                    mean += delta / (i + 1);
                    variance += delta * (value - mean);
                }
                variance /= (data->nmr_samples);
                
                double s_hat = sqrt(variance / data->mu2);
                double l_hat = mean - s_hat * data->mu;
                
                if(!isfinite(l_hat)){
                    l_hat = 0;
                }
                if(!(isfinite(s_hat) && (0 < s_hat))){
                    s_hat = 1;
                }
                
                *(data->means) = l_hat;
                *(data->stds) = s_hat;
            }            
        '''
        return SimpleNamedCLFunction(func, 'compute')
