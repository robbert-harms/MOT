from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import SimpleNamedCLFunction, KernelInputArray, KernelInputScalar, KernelInputAllocatedOutput
from ...cl_routines.base import CLRoutine
import numpy as np

__author__ = 'Robbert Harms'
__date__ = '2017-09-11'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class CircularGaussianFit(CLRoutine):

    def calculate(self, samples, high=np.pi, low=0):
        """Calculate the circular mean and standard deviation wrapped around PI.

        Args:
            samples (ndarray): the samples from which we want to calculate the mean and std.
            low (float): the low boundary for circular mean range
            high (float): the high boundary for circular mean range

        Returns:
            tuple: mean and std arrays
        """
        all_kernel_data = {'samples': KernelInputArray(samples, 'mot_float_type'),
                           'means': KernelInputAllocatedOutput(samples.shape[0], 'mot_float_type', is_readable=False),
                           'stds': KernelInputAllocatedOutput(samples.shape[0], 'mot_float_type', is_readable=False),
                           'nmr_samples': KernelInputScalar(samples.shape[1]),
                           'low': KernelInputScalar(low),
                           'high': KernelInputScalar(high),
                           }

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._get_wrapped_function(), all_kernel_data, samples.shape[0],
                             double_precision=(samples.dtype == np.float64))

        return all_kernel_data['means'].get_data(), all_kernel_data['stds'].get_data()

    def _get_wrapped_function(self):
        func = '''
            void compute(mot_data_struct* data){
                double cos_mean = 0;
                double sin_mean = 0;
                double ang;
                
                for(uint i = 0; i < data->nmr_samples; i++){
                    ang = (data->samples[i] - data->low)*2*M_PI / (data->high - data->low);
                    
                    cos_mean += (cos(ang) - cos_mean) / (i + 1);
                    sin_mean += (sin(ang) - sin_mean) / (i + 1);
                }
                
                double R = hypot(cos_mean, sin_mean);
                if(R > 1){
                    R = 1;
                }
                
                double stds = 1/2. * sqrt(-2 * log(R));
                
                double res = atan2(sin_mean, cos_mean);
                if(res < 0){
                     res += 2 * M_PI;
                }

                *(data->means) = res*(data->high - data->low)/2.0/M_PI + data->low;
                *(data->stds) = ((data->high - data->low)/2.0/M_PI) * sqrt(-2*log(R));
            }            
        '''
        return SimpleNamedCLFunction(func, 'compute')
