import time
import numpy as np
import pyopencl as cl
from ...cl_functions import RanluxCL
from ...utils import get_write_only_cl_mem_flags, initialize_ranlux
from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import Worker


__author__ = 'Robbert Harms'
__date__ = "2014-10-29"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GenerateRandom(AbstractCLRoutine):

    def __init__(self, cl_environments, load_balancer):
        """This class is there to generate random numbers using OpenCL.

        It's main purpose is to verify the correct working of the ranlux random number generator.
        """
        super(GenerateRandom, self).__init__(cl_environments, load_balancer)

    def generate_uniform(self, nmr_samples, minimum, maximum, seed=None):
        """Draw random samples from the uniform distribution.

        Args:
            nmr_samples (int): The number of samples to draw
            minimum (double): The minimum value of the random numbers
            maximum (double): The minimum value of the random numbers
            seed (int): the seed to use, defaults to the number of samples / current time.

        Returns:
            ndarray: A numpy array with nmr_samples random samples drawn from the uniform distribution.
        """
        seed = seed or nmr_samples / time.time()
        return self._generate_samples(nmr_samples, self._get_uniform_kernel(minimum, maximum), seed)

    def generate_gaussian(self, nmr_samples, mean, std, seed=None):
        """Draw random samples from the Gaussian distribution.

        Args:
            nmr_samples (int): The number of samples to draw
            mean (double): The mean of the distribution
            std (double): The standard deviation or the distribution
            seed (int): the seed to use, defaults to the number of samples / current time.

        Returns:
            ndarray: A numpy array with nmr_samples random samples drawn from the Gaussian distribution.
        """
        seed = seed or nmr_samples / time.time()
        return self._generate_samples(nmr_samples, self._get_gaussian_kernel(mean, std), seed)

    def _generate_samples(self, nmr_samples, kernel_source, seed):
        padding = (4 - (nmr_samples % 4)) % 4
        nmr_samples += padding
        samples = np.zeros((nmr_samples + padding,), dtype=np.float32)

        workers = self._create_workers(_GenerateRandomWorker, samples, nmr_samples, kernel_source, seed)
        self.load_balancer.process(workers, nmr_samples / 4)

        if padding:
            return samples[:nmr_samples-padding]
        return samples

    def _get_uniform_kernel(self, min, max):
        kernel_source = '#define RANLUXCL_LUX 4' + "\n"
        kernel_source += RanluxCL().get_cl_code()
        kernel_source += '''
            __kernel void sample(global float4 *ranluxcltab, global float *samples){
                ranluxcl_state_t ranluxclstate;
                ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

                float4 randomnr = ranluxcl(&ranluxclstate);

                int gid = get_global_id(0);

                samples[gid * 4] = ''' + repr(min) + ''' + randomnr.x * ''' + repr(max - min) + ''';
                samples[gid * 4 + 1] = ''' + repr(min) + ''' + randomnr.y * ''' + repr(max - min) + ''';
                samples[gid * 4 + 2] = ''' + repr(min) + ''' + randomnr.z * ''' + repr(max - min) + ''';
                samples[gid * 4 + 3] = ''' + repr(min) + ''' + randomnr.w * ''' + repr(max - min) + ''';
            }
        '''
        return kernel_source

    def _get_gaussian_kernel(self, mean, std):
        kernel_source = '#define RANLUXCL_LUX 4' + "\n"
        kernel_source += RanluxCL().get_cl_code()
        kernel_source += '''
            __kernel void sample(global float4 *ranluxcltab, global float *samples){
                ranluxcl_state_t ranluxclstate;
                ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

                float4 randomnr = ranluxcl_gaussian4(&ranluxclstate);

                int gid = get_global_id(0);

                samples[gid * 4] = ''' + repr(mean) + ''' + randomnr.x * ''' + repr(std) + ''';
                samples[gid * 4 + 1] = ''' + repr(mean) + ''' + randomnr.y * ''' + repr(std) + ''';
                samples[gid * 4 + 2] = ''' + repr(mean) + ''' + randomnr.z * ''' + repr(std) + ''';
                samples[gid * 4 + 3] = ''' + repr(mean) + ''' + randomnr.w * ''' + repr(std) + ''';
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
        self._kernel = self._build_kernel()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        range_start *= 4
        range_end *= 4

        write_only_flags = get_write_only_cl_mem_flags(self._cl_environment)

        ranluxcltab_buffer = initialize_ranlux(self._cl_environment, self._queue, nmr_problems, seed=self._seed)
        samples_buf = cl.Buffer(self._cl_environment.context, write_only_flags,
                                hostbuf=self._samples[range_start:range_end])

        global_range = (int(nmr_problems), )
        local_range = None

        self._kernel.sample(self._queue, global_range, local_range, ranluxcltab_buffer, samples_buf)
        event = cl.enqueue_copy(self._queue, self._samples[range_start:range_end], samples_buf, is_blocking=False)
        return event

    def _get_kernel_source(self):
        return self._kernel_source