import numpy as np
from mot.cl_routines.base import CLRoutine
from mot.cl_routines.mapping.run_procedure import RunProcedure
from mot.library_functions import Rand123
from mot.utils import NameFunctionTuple, is_scalar
from mot.kernel_data import KernelArray, KernelAllocatedArray

__author__ = 'Robbert Harms'
__date__ = "2014-10-29"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Random123Generator(CLRoutine):

    def __init__(self, **kwargs):
        """Create the random123 basis for generating multiple lists of random numbers.

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

        In this implementation we generate a counter and key automatically from a single seed.

        Args:
            seed (float): the seed, if not given a random seed is used.
        """
        super(Random123Generator, self).__init__(**kwargs)

    def rand(self, nmr_distributions, nmr_samples, min_val=0, max_val=1, ctype='float'):
        """Draw random samples from the Uniform distribution.

        Args:
            nmr_distributions (int): the number of unique continuous_distributions to create
            nmr_samples (int): The number of samples to draw
            minimum (double): The minimum value of the random numbers
            maximum (double): The minimum value of the random numbers
            ctype (str): the C type of the output samples

        Returns:
            ndarray: A two dimensional numpy array as (nmr_distributions, nmr_samples).
        """
        if is_scalar(min_val):
            min_val = np.ones((nmr_distributions, 1)) * min_val
        if is_scalar(max_val):
            max_val = np.ones((nmr_distributions, 1)) * max_val

        kernel_data = {'min_val': KernelArray(min_val),
                       'max_val': KernelArray(max_val)}

        return self._generate_samples(nmr_distributions, nmr_samples, ctype, kernel_data,
                                      self._get_uniform_kernel(nmr_samples, ctype))

    def randn(self, nmr_distributions, nmr_samples, mean=0, std=1, ctype='float'):
        """Draw random samples from the Gaussian distribution.

        Args:
            nmr_distributions (int): the number of unique continuous_distributions to create
            nmr_samples (int): The number of samples to draw
            mean (float or ndarray): The mean of the distribution
            std (float or ndarray): The standard deviation or the distribution
            ctype (str): the C type of the output samples

        Returns:
            ndarray: A two dimensional numpy array as (nmr_distributions, nmr_samples).
        """
        if is_scalar(mean):
            mean = np.ones((nmr_distributions, 1)) * mean
        if is_scalar(std):
            std = np.ones((nmr_distributions, 1)) * std

        kernel_data = {'mean': KernelArray(mean),
                       'std': KernelArray(std)}

        return self._generate_samples(nmr_distributions, nmr_samples, ctype, kernel_data,
                                      self._get_gaussian_kernel(nmr_samples, ctype))

    def _generate_samples(self, nmr_distributions, nmr_samples, ctype, kernel_data, cl_function):
        rng_state = np.random.uniform(low=np.iinfo(np.uint32).min, high=np.iinfo(np.uint32).max + 1,
                                      size=(nmr_distributions, 6)).astype(np.uint32)

        kernel_data.update({'samples': KernelAllocatedArray((nmr_distributions, nmr_samples), ctype),
                            '_rng_state': KernelArray(rng_state, 'uint')})

        runner = RunProcedure(self._cl_runtime_info)
        runner.run_procedure(cl_function, kernel_data, nmr_distributions)

        return kernel_data['samples'].get_data()

    def _get_uniform_kernel(self, nmr_samples, ctype):
        random_library = Rand123()
        src = random_library.get_cl_code()
        src += '''
            void compute(mot_data_struct* data){
                rand123_data rand123_rng_data = rand123_initialize_data((uint[]){
                    data->_rng_state[0], data->_rng_state[1], data->_rng_state[2], data->_rng_state[3], 
                    data->_rng_state[4], data->_rng_state[5], 0});
                void* rng_data = (void*)&rand123_rng_data;

                for(uint i = 0; i < ''' + str(nmr_samples) + '''; i++){
                    double4 randomnr = rand4(rng_data);
                    data->samples[i] = (''' + ctype + ''')(data->min_val[0] + 
                                                           randomnr.x * (data->max_val[0] - data->min_val[0]));
                }
            }
        '''
        return NameFunctionTuple('compute', src)

    def _get_gaussian_kernel(self, nmr_samples, ctype):
        random_library = Rand123()
        src = random_library.get_cl_code()
        src += '''
            void compute(mot_data_struct* data){
                rand123_data rand123_rng_data = rand123_initialize_data((uint[]){
                    data->_rng_state[0], data->_rng_state[1], data->_rng_state[2], data->_rng_state[3], 
                    data->_rng_state[4], data->_rng_state[5], 0});
                void* rng_data = (void*)&rand123_rng_data;
                
                for(uint i = 0; i < ''' + str(nmr_samples) + '''; i++){
                    double4 randomnr = randn4(rng_data);
                    data->samples[i] = (''' + ctype + ''')(data->mean[0] + randomnr.x * data->std[0]);
                }
            }
        '''
        return NameFunctionTuple('compute', src)
