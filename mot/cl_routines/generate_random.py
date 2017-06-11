from random import Random

import numpy as np
import pyopencl as cl

from mot.cl_routines.base import CLRoutine
from mot.load_balance_strategies import Worker
from mot.library_functions import Rand123


__author__ = 'Robbert Harms'
__date__ = "2014-10-29"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def generate_uniform(nmr_samples, minimum=0, maximum=1, dtype=None, seed=None):
    """Draw random samples from the uniform distribution.

    Args:
        nmr_samples (int): The number of samples to draw
        minimum (double): The minimum value of the random numbers
        maximum (double): The minimum value of the random numbers
        dtype (np.dtype): the numpy datatype, either one of float32 (default) or float64.
        seed (float): the seed, if not given a random seed is used.

    Returns:
        ndarray: A numpy array with nmr_samples random samples drawn from the uniform distribution.
    """
    generator = Random123GeneratorBase(seed=seed)
    return generator.generate_uniform(nmr_samples, minimum=minimum, maximum=maximum, dtype=dtype)


def generate_gaussian(nmr_samples, mean=0, std=1, dtype=None, seed=None):
    """Draw random samples from the Gaussian distribution.

    Args:
        nmr_samples (int): The number of samples to draw
        mean (double): The mean of the distribution
        std (double): The standard deviation or the distribution
        dtype (np.dtype): the numpy datatype, either one of float32 (default) or float64.
        seed (float): the seed, if not given a random seed is used.

    Returns:
        ndarray: A numpy array with nmr_samples random samples drawn from the Gaussian distribution.
    """
    generator = Random123GeneratorBase(seed=seed)
    return generator.generate_gaussian(nmr_samples, mean=mean, std=std, dtype=dtype)


class Random123GeneratorBase(CLRoutine):

    def __init__(self, seed=None, **kwargs):
        """Create the random123 basis for generating a list of random numbers.

        *From the Random123 documentation:*

        Unlike conventional RNGs, counter-based RNGs are stateless functions (or function classes i.e. functors)
        whose arguments are a counter, and a key and returns a result of the same type as the counter.

        .. code-block:: c

            result = CBRNGname(counter, key)

        The returned result is a deterministic function of the key and counter, i.e. a unique (counter, key)
        tuple will always produce the same result. The result is highly sensitive to small changes in the inputs,
        so that the sequence of values produced by simply incrementing the counter (or key) is effectively
        indistinguishable from a sequence of samples of a uniformly distributed random variable.

        All the Random123 generators are counter-based RNGs that use integer multiplication, xor and permutation
        of W-bit words to scramble its N-word input key.


        *Implementation note*:

        In this implementation we generate a counter and key automatically from a single seed.

        Args:
            seed (float): the seed, if not given a random seed is used.
        """
        super(Random123GeneratorBase, self).__init__(**kwargs)
        self.context = self.cl_environments[0].get_cl_context().context
        self._rng_state = self._get_rng_state(seed)

    def _get_rng_state(self, seed):
        if seed is None:
            seed = Random().randint(0, 2 ** 31)

        rng = Random(seed)
        dtype_info = np.iinfo(np.uint32)

        return np.array(list(rng.randrange(dtype_info.min, dtype_info.max + 1) for _ in range(6)), dtype=np.uint32)

    def generate_uniform(self, nmr_samples, minimum=0, maximum=1, dtype=None):
        """Draw random samples from the uniform distribution.

        Args:
            nmr_samples (int): The number of samples to draw
            minimum (double): The minimum value of the random numbers
            maximum (double): The minimum value of the random numbers
            dtype (np.dtype): the numpy datatype, either one of float32 (default) or float64.

        Returns:
            ndarray: A numpy array with nmr_samples random samples drawn from the uniform distribution.
        """
        dtype = dtype or np.float32
        if dtype not in (np.float32, np.float64):
            raise ValueError('The given dtype should be either float32 or float64, {} given.'.format(
                dtype.__class__.__name__))

        c_type = 'float'
        if dtype == np.float64:
            c_type = "double"

        return self._generate_samples(nmr_samples, self._get_uniform_kernel(minimum, maximum, c_type))

    def generate_gaussian(self, nmr_samples, mean=0, std=1, dtype=None):
        """Draw random samples from the Gaussian distribution.

        Args:
            nmr_samples (int): The number of samples to draw
            mean (double): The mean of the distribution
            std (double): The standard deviation or the distribution
            dtype (np.dtype): the numpy datatype, either one of float32 (default) or float64.

        Returns:
            ndarray: A numpy array with nmr_samples random samples drawn from the Gaussian distribution.
        """
        dtype = dtype or np.float32
        if dtype not in (np.float32, np.float64):
            raise ValueError('The given dtype should be either float32 or float64, {} given.'.format(
                dtype.__class__.__name__))

        c_type = 'float'
        if dtype == np.float64:
            c_type = "double"

        return self._generate_samples(nmr_samples, self._get_gaussian_kernel(mean, std, c_type))

    def _generate_samples(self, nmr_samples, kernel_source):
        padding = (-nmr_samples) % 4
        nmr_samples += padding
        samples = np.zeros((nmr_samples,), dtype=np.float32)

        workers = self._create_workers(lambda cl_environment: _Random123Worker(cl_environment, samples,
                                                                               kernel_source, self._rng_state))
        self.load_balancer.process(workers, nmr_samples // 4)

        if padding:
            return samples[:-padding]
        return samples

    def _get_uniform_kernel(self, min_val, max_val, c_type):
        random_library = Rand123()
        src = random_library.get_cl_code()
        # By setting the rand123 state as kernel arguments the kernel does not need to be recompiled for a new state.
        src += '''
            __kernel void generate(constant uint* rng_state,
                                   global ''' + c_type + '''* samples){

                rand123_data rng_data = rand123_initialize_data(
                    (uint[]){rng_state[0], rng_state[1], rng_state[2], rng_state[3], rng_state[4], rng_state[5],
                             get_global_id(0), 0});

                ''' + c_type + '''4 randomnr =  rand123_uniform_''' + c_type + '''4(&rng_data);

                ulong gid = get_global_id(0);

                samples[gid * 4] = ''' + str(min_val) + ''' + randomnr.x * ''' + str(max_val - min_val) + ''';
                samples[gid * 4 + 1] = ''' + str(min_val) + ''' + randomnr.y * ''' + str(max_val - min_val) + ''';
                samples[gid * 4 + 2] = ''' + str(min_val) + ''' + randomnr.z * ''' + str(max_val - min_val) + ''';
                samples[gid * 4 + 3] = ''' + str(min_val) + ''' + randomnr.w * ''' + str(max_val - min_val) + ''';
            }
        '''
        return src

    def _get_gaussian_kernel(self, mean, std, c_type):
        random_library = Rand123()
        src = random_library.get_cl_code()
        # By setting the rand123 state as kernel arguments the kernel does not need to be recompiled for a new state.
        src += '''
            __kernel void generate(constant uint* rng_state,
                                   global ''' + c_type + '''* samples){

                rand123_data rng_data = rand123_initialize_data(
                    (uint[]){rng_state[0], rng_state[1], rng_state[2], rng_state[3], rng_state[4], rng_state[5],
                             get_global_id(0), 0});

                ''' + c_type + '''4 randomnr =  rand123_normal_''' + c_type + '''4(&rng_data);

                ulong gid = get_global_id(0);

                samples[gid * 4] = ''' + str(mean) + ''' + randomnr.x * ''' + str(std) + ''';
                samples[gid * 4 + 1] = ''' + str(mean) + ''' + randomnr.y * ''' + str(std) + ''';
                samples[gid * 4 + 2] = ''' + str(mean) + ''' + randomnr.z * ''' + str(std) + ''';
                samples[gid * 4 + 3] = ''' + str(mean) + ''' + randomnr.w * ''' + str(std) + ''';
            }
        '''
        return src


class _Random123Worker(Worker):

    def __init__(self, cl_environment, samples, kernel_source, rng_state):
        super(_Random123Worker, self).__init__(cl_environment)
        self._samples = samples
        self._nmr_samples = self._samples.shape[0]
        self._kernel_source = kernel_source
        self._rng_state = rng_state

        self._samples_buf = cl.Buffer(self._cl_run_context.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                      hostbuf=self._samples)

        self._rng_state_buffer = cl.Buffer(self._cl_run_context.context,
                                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._rng_state)

        self._kernel = self._build_kernel(self._get_kernel_source())

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start
        kernel_args = [self._rng_state_buffer, self._samples_buf]
        self._kernel.generate(self._cl_run_context.queue, (int(nmr_problems), ), None,
                              *kernel_args, global_offset=(range_start,))
        self._enqueue_readout(self._samples_buf, self._samples, range_start * 4, range_end * 4)

    def _get_kernel_source(self):
        return self._kernel_source
