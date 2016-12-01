import os
import time
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from mot.utils import initialize_ranlux, get_ranlux_cl, get_random123_cl_code
from mot.cl_routines.base import CLRoutine
from mot.load_balance_strategies import Worker
from pkg_resources import resource_filename
from random import Random

__author__ = 'Robbert Harms'
__date__ = "2014-10-29"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class RanluxRandom(CLRoutine):

    def __init__(self, cl_environments, load_balancer):
        """Generate random numbers using the Ranlux RNG.
        """
        super(RanluxRandom, self).__init__(cl_environments, load_balancer)

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

        workers = self._create_workers(lambda cl_environment: _RanluxRandomWorker(cl_environment, samples,
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


class _RanluxRandomWorker(Worker):

    def __init__(self, cl_environment, samples, nmr_samples, kernel_source, seed):
        super(_RanluxRandomWorker, self).__init__(cl_environment)
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


class Random123StartingPoint():

    def get_key(self, dtype):
        """Gets the key used as starting point for the Random123 generator.

        This should either return 2 or 4 keys depending on the desired precision.

        Args:
            dtype (np.dtype): the datatype to use for the key

        Returns:
            list: the key used as starting point in the Random123 generator
        """

    def get_counter(self, dtype, length):
        """Gets the counter used as starting point for the Random123 generator.

        Args:
            dtype (np.dtype): the datatype to use
            length (int): the length of the returned list

        Returns:
            list: the counter used as starting point in the Random123 generator
        """


class StartingPointFromSeed(Random123StartingPoint):

    def __init__(self, seed, key_length=None):
        """Generates the key and counter from the given seed.

        Args:
            seed (int): the seed used to generate a starting point for the Random123 RNG.
            key_length (int): the length of the key, either 2 or 4 depending on the desired precision.
        """
        self._seed = seed
        self._key_length = key_length
        if self._key_length not in (2, 4):
            self._key_length = 4

    def get_key(self, dtype):
        rng = Random(self._seed)
        dtype_info = np.iinfo(dtype)
        key = [rng.randrange(dtype_info.min, dtype_info.max + 1) for _ in range(self._key_length - 2)]
        return key

    def get_counter(self, dtype, length):
        rng = Random(self._seed)
        dtype_info = np.iinfo(dtype)

        counter = [rng.randrange(dtype_info.min, dtype_info.max + 1) for _ in range(length)]

        return counter


class Random123GeneratorBase(CLRoutine):

    def __init__(self, starting_point=None, generator_name=None, nmr_words=None, word_bitsize=None, **kwargs):
        """Create the random123 basis for generating a list of random numbers.

        *From the Random123 documentation:*

        Unlike conventional RNGs, counter-based RNGs are stateless functions (or function classes i.e. functors)
        whose arguments are a counter, and a key and returns a result of the same type as the counter.

        .. code-block: c

            result = CBRNGname(counter, key)

        The returned result is a deterministic function of the key and counter, i.e. a unique (counter, key)
        tuple will always produce the same result. The result is highly sensitive to small changes in the inputs,
        so that the sequence of values produced by simply incrementing the counter (or key) is effectively
        indistinguishable from a sequence of samples of a uniformly distributed random variable.

        All the Random123 generators are counter-based RNGs that use integer multiplication, xor and permutation
        of W-bit words to scramble its N-word input key.

        Args:
            starting_point (Random123StartingPoint): object that provides the generator with the starting points
                (key, counter) tuple. If None is given we will use the ``StartingPointFromSeed`` with a seed of 0.
            generator_name (str): the generator to use, either 'threefry' or 'philox'. Defaults to 'threefry'.
            nmr_words (int): the number of words used in the input key and counter. Defaults to 4.
            word_bitsize (int): the size of the bit-words used as key and counter input, either 32 or 64.
                Defaults to 32.
        """
        super(Random123GeneratorBase, self).__init__(**kwargs)

        starting_point = starting_point or StartingPointFromSeed(0)

        self.context = self.cl_environments[0].get_cl_context().context

        self._generator_name = generator_name or 'threefry'
        self._nmr_words = nmr_words or 4
        self._word_bitsize = word_bitsize or 32

        if self._nmr_words not in (2, 4):
            raise ValueError('The number of words should be either 2 or 4, {} given.'.format(self._nmr_words))

        if self._word_bitsize not in (32, 64):
            raise ValueError('The word bitsize should be either 32 or 64, {} given.'.format(self._word_bitsize))

        if self._word_bitsize == 32:
            dtype = np.uint32
        else:
            dtype = np.uint64

        self._key = np.array(starting_point.get_key(dtype), dtype=dtype)
        self._key_length = len(self._key) + 2
        self._counter = np.array(starting_point.get_counter(dtype, self._nmr_words), dtype=dtype)

    def _get_uniform_kernel(self, min_val, max_val, c_type):
        src = get_random123_cl_code(self._generator_name, self._nmr_words, self._word_bitsize)
        src += '''
            __kernel void generate(constant uint* rand123_counter,
                                   ''' + ('constant uint* rand123_key,' if self._key_length > 2 else '') + '''
                                   global ''' + c_type + '''* samples){

                rand123_data rng_data = ''' + \
                        ('rand123_initialize_data_4key_constmem(rand123_counter, rand123_key)' if self._key_length > 2
                         else 'rand123_initialize_data_2key_constmem(rand123_counter)') + ''';
                rand123_set_loop_key(&rng_data, 0);

                ''' + c_type + '''4 randomnr =  rand123_uniform_''' + c_type + '''4(&rng_data);

                int gid = get_global_id(0);

                samples[gid * 4] = ''' + str(min_val) + ''' + randomnr.x * ''' + str(max_val - min_val) + ''';
                samples[gid * 4 + 1] = ''' + str(min_val) + ''' + randomnr.y * ''' + str(max_val - min_val) + ''';
                samples[gid * 4 + 2] = ''' + str(min_val) + ''' + randomnr.z * ''' + str(max_val - min_val) + ''';
                samples[gid * 4 + 3] = ''' + str(min_val) + ''' + randomnr.w * ''' + str(max_val - min_val) + ''';
            }
        '''
        return src

    def _get_gaussian_kernel(self, mean, std, c_type):
        src = get_random123_cl_code(self._generator_name, self._nmr_words, self._word_bitsize)
        src += '''
            __kernel void generate(constant uint* rand123_counter,
                                   ''' + ('constant uint* rand123_key,' if self._key_length > 2 else '') + '''
                                   global ''' + c_type + '''* samples){

                rand123_data rng_data = ''' + \
                        ('rand123_initialize_data_4key_constmem(rand123_counter, rand123_key)' if self._key_length > 2
                         else 'rand123_initialize_data_2key_constmem(rand123_counter)') + ''';
                rand123_set_loop_key(&rng_data, 0);

                ''' + c_type + '''4 randomnr =  rand123_normal_''' + c_type + '''4(&rng_data);

                int gid = get_global_id(0);

                samples[gid * 4] = ''' + str(mean) + ''' + randomnr.x * ''' + str(std) + ''';
                samples[gid * 4 + 1] = ''' + str(mean) + ''' + randomnr.y * ''' + str(std) + ''';
                samples[gid * 4 + 2] = ''' + str(mean) + ''' + randomnr.z * ''' + str(std) + ''';
                samples[gid * 4 + 3] = ''' + str(mean) + ''' + randomnr.w * ''' + str(std) + ''';
            }
        '''
        return src

    def _get_kernel_source(self, dtype, distribution):
        c_type = 'float'
        if dtype == np.float64:
            c_type = "double"

        if distribution == 'normal':
            return self._get_gaussian_kernel(0, 1, c_type)
        else:
            return self._get_uniform_kernel(0, 1, c_type)

    def get_gen_kernel(self, dtype, distribution):
        src = self._get_kernel_source(dtype, distribution)

        prg = cl.Program(self.context, src).build()
        knl = prg.generate

        return knl

    def _fill(self, distribution, ary, queue=None):
        """Fill *ary* with uniformly distributed random numbers in the interval
        *(a, b)*, endpoints excluded.

        :return: a :class:`pyopencl.Event`
        """
        if queue is None:
            queue = ary.queue

        knl = self.get_gen_kernel(ary.dtype, distribution)
        args = []

        counter_buffer = cl.Buffer(self.context,
                                   cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                   hostbuf=self._counter)

        args.append(counter_buffer)

        if self._key_length > 2:
            key_buffer = cl.Buffer(self.context,
                                   cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                   hostbuf=self._key)
            args.append(key_buffer)

        args.append(ary.data)
        return knl(queue, (ary.shape[0]//4,), None, *args)

    def fill_uniform(self, ary, queue=None):
        return self._fill("uniform", ary, queue=queue)

    def uniform(self, *args, **kwargs):
        """Make a new empty array, apply :meth:`fill_uniform` to it.
        """
        result = cl_array.empty(*args, **kwargs)
        result.add_event(self.fill_uniform(result, queue=result.queue))
        return result

    def fill_normal(self, ary, queue=None):
        """Fill *ary* with normally distributed numbers with mean *mu* and
        standard deviation *sigma*.
        """
        return self._fill("normal", ary, queue=queue)

    def normal(self, *args, **kwargs):
        """Make a new empty array, apply :meth:`fill_normal` to it.
        """
        result = cl_array.empty(*args, **kwargs)
        result.add_event(self.fill_normal(result, queue=result.queue))
        return result


def rand(cl_environments, shape, dtype, start=0, stop=1):
    """Return an array of `shape` filled with random values of `dtype` in the range [start, stop).
    """

    gen = Random123GeneratorBase(cl_environments=cl_environments)

    from pyopencl.array import Array
    result = Array(cl_environments[0].get_cl_context().queue, shape, dtype)
    # result.add_event(gen.fill_uniform(result))
    result.add_event(gen.fill_normal(result))
    return result.get()
