import numpy as np
import pyopencl as cl
from mot.utils import get_random123_cl_code
from mot.cl_routines.base import CLRoutine
from mot.load_balance_strategies import Worker
from random import Random

__author__ = 'Robbert Harms'
__date__ = "2014-10-29"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Random123StartingPoint(object):

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
        self._key_length = key_length or 4
        if self._key_length not in (2, 4):
            raise ValueError('The key length should be either 2 or 4, {} given.'.format(key_length))

    def get_key(self, dtype):
        rng = Random(self._seed)
        dtype_info = np.iinfo(dtype)
        key = [rng.randrange(dtype_info.min, dtype_info.max + 1) for _ in range(self._key_length - 2)]
        return key

    def get_counter(self, dtype, length):
        rng = Random(self._seed + 1)
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
                (key, counter) tuple. If None is given we will use the ``StartingPointFromSeed`` with a random seed.
            generator_name (str): the generator to use, either 'threefry' or 'philox'. Defaults to 'threefry'.
            nmr_words (int): the number of words used in the input key and counter. Defaults to 4.
            word_bitsize (int): the size of the bit-words used as key and counter input, either 32 or 64.
                Defaults to 32.
        """
        super(Random123GeneratorBase, self).__init__(**kwargs)

        starting_point = starting_point or StartingPointFromSeed(Random().randint(0, 1e6))

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
                                                                               kernel_source, self._key, self._counter))
        self.load_balancer.process(workers, nmr_samples // 4)

        if padding:
            return samples[:-padding]
        return samples

    def _get_rand123_init_cl_code(self):
        if self._key_length > 2:
            return 'rand123_initialize_data_4key_constmem(rand123_counter, rand123_key)'
        else:
            return 'rand123_initialize_data_2key_constmem(rand123_counter)'

    def _get_uniform_kernel(self, min_val, max_val, c_type):
        src = get_random123_cl_code(self._generator_name, self._nmr_words, self._word_bitsize)
        src += '''
            __kernel void generate(constant uint* rand123_counter,
                                   ''' + ('constant uint* rand123_key,' if self._key_length > 2 else '') + '''
                                   global ''' + c_type + '''* samples){

                rand123_data rng_data = ''' + self._get_rand123_init_cl_code() + ''';
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

                rand123_data rng_data = ''' + self._get_rand123_init_cl_code() + ''';
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


class _Random123Worker(Worker):

    def __init__(self, cl_environment, samples, kernel_source, key, counter):
        super(_Random123Worker, self).__init__(cl_environment)
        self._samples = samples
        self._nmr_samples = self._samples.shape[0]
        self._kernel_source = kernel_source
        self._key = key
        self._counter = counter

        self._samples_buf = cl.Buffer(self._cl_run_context.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                      hostbuf=self._samples)

        self._counter_buffer = cl.Buffer(self._cl_run_context.context,
                                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._counter)

        if len(self._key):
            self._key_buffer = cl.Buffer(self._cl_run_context.context,
                                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self._key)

        self._kernel = self._build_kernel()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        kernel_args = [self._counter_buffer]
        if len(self._key):
            kernel_args.append(self._key_buffer)
        kernel_args.append(self._samples_buf)

        event = self._kernel.generate(self._cl_run_context.queue, (int(nmr_problems), ), None,
                                      *kernel_args, global_offset=(range_start,))
        return [self._enqueue_readout(self._samples_buf, self._samples, range_start * 4, range_end * 4, [event])]

    def _get_kernel_source(self):
        return self._kernel_source
