import time
import numpy as np
import pyopencl as cl
from ...utils import initialize_ranlux, get_ranlux_cl
from ...cl_routines.base import CLRoutine
from ...load_balance_strategies import Worker


__author__ = 'Robbert Harms'
__date__ = "2014-10-29"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GenerateRandom(CLRoutine):

    def __init__(self, cl_environments, load_balancer):
        """This class is there to generate random numbers using OpenCL.

        It's main purpose is to verify the correct working of the ranlux random number generator.
        """
        super(GenerateRandom, self).__init__(cl_environments, load_balancer)

    def generate_uniform(self, nmr_samples, minimum=0, maximum=1, seed=None):
        """Draw random samples from the uniform distribution.

        Args:
            nmr_samples (int): The number of samples to draw
            minimum (double): The minimum value of the random numbers
            maximum (double): The minimum value of the random numbers
            seed (int): the seed to use, defaults to the number of samples / current time.

        Returns:
            ndarray: A numpy array with nmr_samples random samples drawn from the uniform distribution.
        """
        seed = seed or (nmr_samples / time.time()) * 1e10
        return self._generate_samples(nmr_samples, self._get_uniform_kernel(minimum, maximum), seed)

    def generate_gaussian(self, nmr_samples, mean=0, std=1, seed=None):
        """Draw random samples from the Gaussian distribution.

        Args:
            nmr_samples (int): The number of samples to draw
            mean (double): The mean of the distribution
            std (double): The standard deviation or the distribution
            seed (int): the seed to use, defaults to the number of samples / current time.

        Returns:
            ndarray: A numpy array with nmr_samples random samples drawn from the Gaussian distribution.
        """
        seed = seed or (nmr_samples / time.time()) * 1e10
        return self._generate_samples(nmr_samples, self._get_gaussian_kernel(mean, std), seed)

    def _generate_samples(self, nmr_samples, kernel_source, seed):
        padding = (4 - (nmr_samples % 4)) % 4
        nmr_samples += padding
        samples = np.zeros((nmr_samples + padding,), dtype=np.float32)

        workers = self._create_workers(lambda cl_environment: _GenerateRandomWorker(cl_environment, samples,
                                                                                    nmr_samples, kernel_source, seed))
        self.load_balancer.process(workers, nmr_samples / 4)

        if padding:
            return samples[:nmr_samples-padding]
        return samples

    def _get_uniform_kernel(self, min_val, max_val):
        kernel_source = '#define RANLUXCL_LUX 4' + "\n"
        kernel_source += get_ranlux_cl()
        kernel_source += '''
            __kernel void sample(global ranluxcl_state_t *ranluxcltab, global float *samples){
                ranluxcl_state_t ranluxclstate;
                ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

                float4 randomnr = ranluxcl32(&ranluxclstate);

                int gid = get_global_id(0);

                samples[gid * 4] = ''' + str(min_val) + ''' + randomnr.x * ''' + str(max_val - min_val) + ''';
                samples[gid * 4 + 1] = ''' + str(min_val) + ''' + randomnr.y * ''' + str(max_val - min_val) + ''';
                samples[gid * 4 + 2] = ''' + str(min_val) + ''' + randomnr.z * ''' + str(max_val - min_val) + ''';
                samples[gid * 4 + 3] = ''' + str(min_val) + ''' + randomnr.w * ''' + str(max_val - min_val) + ''';

                ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
            }
        '''
        return kernel_source

    def _get_gaussian_kernel(self, mean, std):
        kernel_source = '#define RANLUXCL_LUX 4' + "\n"
        kernel_source += get_ranlux_cl()
        kernel_source += '''
            __kernel void sample(global ranluxcl_state_t *ranluxcltab, global float *samples){
                ranluxcl_state_t ranluxclstate;
                ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

                float4 randomnr = ranluxcl_gaussian4(&ranluxclstate);

                int gid = get_global_id(0);

                samples[gid * 4] = ''' + str(mean) + ''' + randomnr.x * ''' + str(std) + ''';
                samples[gid * 4 + 1] = ''' + str(mean) + ''' + randomnr.y * ''' + str(std) + ''';
                samples[gid * 4 + 2] = ''' + str(mean) + ''' + randomnr.z * ''' + str(std) + ''';
                samples[gid * 4 + 3] = ''' + str(mean) + ''' + randomnr.w * ''' + str(std) + ''';

                ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
            }
        '''
        return kernel_source


class _GenerateRandomWorker(Worker):

    def __init__(self, cl_environment, samples, nmr_samples, kernel_source, seed):
        super(_GenerateRandomWorker, self).__init__(cl_environment)
        self._samples = samples
        self._nmr_samples = nmr_samples
        self._kernel_source = kernel_source
        self._seed = seed

        self._ranluxcltab_buffer = initialize_ranlux(self._cl_run_context, self._nmr_samples, seed=self._seed)

        self._samples_buf = cl.Buffer(self._cl_run_context.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                      hostbuf=self._samples)

        self._kernel = self._build_kernel()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        event = self._kernel.sample(self._cl_run_context.queue, (int(nmr_problems), ), None,
                                    self._ranluxcltab_buffer, self._samples_buf, global_offset=(range_start,))
        return [self._enqueue_readout(self._samples_buf, self._samples, range_start * 4, range_end * 4, [event])]

    def _get_kernel_source(self):
        return self._kernel_source
