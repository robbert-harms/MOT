import os
import time
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from mot.utils import initialize_ranlux, get_ranlux_cl
from mot.cl_routines.base import CLRoutine
from mot.load_balance_strategies import Worker
from pkg_resources import resource_filename

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


class Random123GeneratorBase(CLRoutine):

    def __init__(self, key=None, counter=None, seed=None, cl_environments=None, load_balancer=None):
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

        Args:
            key (list of int): a list of integers to use for the key, the length should be ``key_length - 1``
                The range of each int should be between the minimum and the maximum allowed integer on this system, you
                can find these using np.iinfo(np.int32).
            counter (list of int): a list of integers to use for the counter, of length 4.
                The range of each int should be between the minimum and the maximum allowed integer on this system, you
                can find these using np.iinfo(np.int32).
            seed (int): the seed to use to generate a random key and counter array if these are not provided.
        """
        super(Random123GeneratorBase, self).__init__(cl_environments, load_balancer)

        if all(v is not None for v in (seed, key, counter)):
            raise TypeError("If both counter and key are given, seed can not be given.")

        int32_info = np.iinfo(np.int32)

        if key is None:
            key = np.random.randint(int(int32_info.min), high=int(int32_info.max) + 1,
                                    size=self._get_key_length()-1, dtype=np.int32).tolist()
        if counter is None:
            counter = np.random.randint(int(int32_info.min), high=int(int32_info.max)+1,
                                        size=4, dtype=np.int32).tolist()

        self.context = self.cl_environments[0].get_cl_context().context
        self.key = key
        self.counter = counter

        self.counter_max = int32_info.max

    def _get_key_length(self):
        """Get the key length of the implementing Random123 generator.

        Returns:
            int: the key length for the generator in use.
        """
        raise NotImplementedError()

    def _get_rng_settings(self, distribution, dtype):
        size_multiplier = 1
        arg_dtype = dtype

        if distribution not in ('uniform', 'normal'):
            raise TypeError("Unknown distribution asked ({}), only "
                            "uniform and normal are supported.".format(distribution))

        if dtype == np.float64:
            c_type = "double"

            scale1_const = "((double) %r)" % (1/2**32)
            scale2_const = "((double) %r)" % (1/2**64)

            transform = ''
            if distribution == "normal":
                transform = "box_muller"

            rng_expr = (
                    "shift + scale * "
                    "%s( %s * convert_double4(gen)"
                    "+ %s * convert_double4(gen))"
                    % (transform, scale1_const, scale2_const))

            counter_multiplier = 2
            return c_type, rng_expr, counter_multiplier, size_multiplier, arg_dtype

        if dtype in [np.float32, cl.array.vec.float2, cl.array.vec.float3, cl.array.vec.float4]:
            c_type = "float"
            scale_const = "((float) %r)" % (1/2**32)

            transform = ''
            if distribution == 'normal':
                transform = 'box_muller'

            rng_expr = ("shift + scale * %s(%s * convert_float4(gen))" % (transform, scale_const))
            counter_multiplier = 1
            arg_dtype = np.float32
            try:
                _, size_multiplier = cl.array.vec.type_to_scalar_and_count[dtype]
            except KeyError:
                pass

            return c_type, rng_expr, counter_multiplier, size_multiplier, arg_dtype

        if distribution == 'uniform' and dtype == np.int32:
            c_type = "int"
            rng_expr = (
                    "shift + convert_int4((convert_long4(gen) * scale) / %s)"
                    % (str(2**32)+"l")
                    )
            counter_multiplier = 1
            return c_type, rng_expr, counter_multiplier, size_multiplier, arg_dtype

        if distribution == 'uniform' and dtype == np.int64:
            c_type = "long"
            rng_expr = (
                    "shift"
                    "+ convert_long4(gen) * (scale/two32) "
                    "+ ((convert_long4(gen) * scale) / two32)"
                    .replace("two32", (str(2**32)+"l")))
            counter_multiplier = 2
            return c_type, rng_expr, counter_multiplier, size_multiplier, arg_dtype

        raise TypeError("Unsupported comnbination of distribution and data "
                        "type given: '{}/{}'".format(distribution, dtype))

    def _get_kernel_source(self, dtype, distribution):
        src = open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/openclfeatures.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/array.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/{}.cl'.format(self.header_name)), ),
                    'r').read()

        c_type, rng_expr, counter_multiplier, size_multiplier, arg_dtype = self._get_rng_settings(distribution, dtype)

        src += """
                    typedef %(output_t)s output_t;
                    typedef %(output_t)s4 output_vec_t;
                    typedef %(gen_name)s_ctr_t ctr_t;
                    typedef %(gen_name)s_key_t key_t;

                    uint4 gen_bits(key_t *key, ctr_t *ctr)
                    {
                        union {
                            ctr_t ctr_el;
                            uint4 vec_el;
                        } u;

                        u.ctr_el = %(gen_name)s(*ctr, *key);
                        if (++ctr->v[0] == 0)
                            if (++ctr->v[1] == 0)
                                ++ctr->v[2];

                        return u.vec_el;
                    }

                    #if %(include_box_muller)s
                    output_vec_t box_muller(output_vec_t x)
                    {
                        #define BOX_MULLER(I, COMPA, COMPB) \
                            output_t r##I = sqrt(-2*log(x.COMPA)); \
                            output_t c##I; \
                            output_t s##I = sincos((output_t) (2*M_PI) * x.COMPB, &c##I);

                        BOX_MULLER(0, x, y);
                        BOX_MULLER(1, z, w);
                        return (output_vec_t) (r0*c0, r0*s0, r1*c1, r1*s1);
                    }
                    #endif

                    #define GET_RANDOM_NUM(gen) %(rng_expr)s

                    kernel void generate(
                        int k1,
                        #if %(key_length)s > 2
                        int k2, int k3,
                        #endif
                        int c0, int c1, int c2, int c3,
                        global output_t *output,
                        long out_size,
                        output_t scale,
                        output_t shift)
                    {
                        #if %(key_length)s == 2
                        key_t k = {{get_global_id(0), k1}};
                        #else
                        key_t k = {{get_global_id(0), k1, k2, k3}};
                        #endif

                        ctr_t c = {{c0, c1, c2, c3}};

                        // output bulk
                        unsigned long idx = get_global_id(0)*4;
                        while (idx + 4 < out_size)
                        {
                            *(global output_vec_t *) (output + idx) =
                                GET_RANDOM_NUM(gen_bits(&k, &c));
                            idx += 4*get_global_size(0);
                        }

                        // output tail
                        output_vec_t tail_ran = GET_RANDOM_NUM(gen_bits(&k, &c));
                        if (idx < out_size)
                          output[idx] = tail_ran.x;
                        if (idx+1 < out_size)
                          output[idx+1] = tail_ran.y;
                        if (idx+2 < out_size)
                          output[idx+2] = tail_ran.z;
                        if (idx+3 < out_size)
                          output[idx+3] = tail_ran.w;
                    }
                    """ % {
            "gen_name": self.generator_name,
            "output_t": c_type,
            "key_length": self._get_key_length(),
            "include_box_muller": int(distribution == "normal"),
            "rng_expr": rng_expr
        }

        return src

    def get_gen_kernel(self, dtype, distribution):
        c_type, rng_expr, counter_multiplier, size_multiplier, arg_dtype = self._get_rng_settings(distribution, dtype)

        src = self._get_kernel_source(dtype, distribution)

        prg = cl.Program(self.context, src).build()
        knl = prg.generate
        knl.set_scalar_arg_dtypes(
                [np.int32] * (self._get_key_length() - 1 + 4)
                + [None, np.int64, arg_dtype, arg_dtype])

        return knl, counter_multiplier, size_multiplier

    def _fill(self, distribution, ary, scale, shift, queue=None):
        """Fill *ary* with uniformly distributed random numbers in the interval
        *(a, b)*, endpoints excluded.

        :return: a :class:`pyopencl.Event`
        """

        if queue is None:
            queue = ary.queue

        knl, counter_multiplier, size_multiplier = \
                self.get_gen_kernel(ary.dtype, distribution)

        args = self.key + self.counter + [
                ary.data, ary.size*size_multiplier,
                scale, shift]

        n = ary.size
        from pyopencl.array import splay
        gsize, lsize = splay(queue, ary.size)

        evt = knl(queue, gsize, lsize, *args)

        self.counter[0] += n * counter_multiplier
        c1_incr, self.counter[0] = divmod(self.counter[0], self.counter_max)
        if c1_incr:
            self.counter[1] += c1_incr
            c2_incr, self.counter[1] = divmod(self.counter[1], self.counter_max)
            self.counter[2] += c2_incr

        return evt

    def fill_uniform(self, ary, a=0, b=1, queue=None):
        return self._fill("uniform", ary,
                scale=(b-a), shift=a, queue=queue)

    def uniform(self, *args, **kwargs):
        """Make a new empty array, apply :meth:`fill_uniform` to it.
        """
        a = kwargs.pop("a", 0)
        b = kwargs.pop("b", 1)

        result = cl_array.empty(*args, **kwargs)

        result.add_event(
                self.fill_uniform(result, queue=result.queue, a=a, b=b))
        return result

    def fill_normal(self, ary, mu=0, sigma=1, queue=None):
        """Fill *ary* with normally distributed numbers with mean *mu* and
        standard deviation *sigma*.
        """

        return self._fill("normal", ary, scale=sigma, shift=mu, queue=queue)

    def normal(self, *args, **kwargs):
        """Make a new empty array, apply :meth:`fill_normal` to it.
        """
        mu = kwargs.pop("mu", 0)
        sigma = kwargs.pop("sigma", 1)

        result = cl_array.empty(*args, **kwargs)

        result.add_event(
                self.fill_normal(result, queue=result.queue, mu=mu, sigma=sigma))
        return result


class Random123SourceBuilder(object):

    def __init__(self, dtype, distribution):
        """Construct a random generator meant to generate random numbers for the given data type and distribution type.

        The distributions supported are 'normal' and 'uniform', and with 'normal' only supported for floating point
        data types.

        Args:
            dtype (np.dtype): the numpy datatype we wish our samples to be in

        """


class PhiloxGenerator(Random123GeneratorBase):

    header_name = "philox"
    generator_name = "philox4x32"

    def _get_key_length(self):
        return 2


class ThreefryGenerator(Random123GeneratorBase):

    header_name = "threefry"
    generator_name = "threefry4x32"

    def _get_key_length(self):
        return 4


def rand(cl_environments, shape, dtype, start=0, stop=1):
    """Return an array of `shape` filled with random values of `dtype` in the range [start, stop).
    """
    if cl_environments[0].is_cpu:
        gen = PhiloxGenerator(cl_environments=cl_environments)
    else:
        gen = ThreefryGenerator(cl_environments=cl_environments)

    from pyopencl.array import Array
    result = Array(cl_environments[0].get_cl_context().queue, shape, dtype)
    # result.add_event(gen.fill_uniform(result, a=start, b=stop))
    result.add_event(gen.fill_normal(result))
    return result.get()
