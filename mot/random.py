"""This uses the random123 library for generating multiple lists of random numbers.

*From the Random123 documentation:*

Unlike conventional RNGs, counter-based RNGs are stateless functions (or function classes i.e. functors)
whose arguments are a counter and a key, and returns a result of the same type as the counter.

.. code-block:: c

    result = CBRNGname(counter, key)

The result is producted by a deterministic function of the key and counter, i.e. a unique (counter, key)
tuple will always produce the same result. The result is highly sensitive to small changes in the inputs,
so that the sequence of values produced by simply incrementing the counter (or key) is effectively
indistinguishable from a sequence of samples of a uniformly distributed random variable.

All the Random123 generators are counter-based RNGs that use integer multiplication, xor and permutation
of W-bit words to scramble its N-word input key.

In this implementation we generate a counter and key automatically from a single seed.
"""
import numpy as np

from mot.lib.cl_function import SimpleCLFunction
from mot.library_functions import Rand123
from mot.lib.utils import is_scalar
from mot.lib.kernel_data import Array, Zeros

__author__ = 'Robbert Harms'
__date__ = '2018-08-01'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


def uniform(nmr_distributions, nmr_samples, low=0, high=1, ctype='float', seed=None):
    """Draw random samples from the Uniform distribution.

    Args:
        nmr_distributions (int): the number of unique continuous_distributions to create
        nmr_samples (int): The number of samples to draw
        low (double): The minimum value of the random numbers
        high (double): The minimum value of the random numbers
        ctype (str): the C type of the output samples
        seed (float): the seed for the RNG

    Returns:
        ndarray: A two dimensional numpy array as (nmr_distributions, nmr_samples).
    """
    if is_scalar(low):
        low = np.ones((nmr_distributions, 1)) * low
    if is_scalar(high):
        high = np.ones((nmr_distributions, 1)) * high

    kernel_data = {'min_val': Array(low),
                   'max_val': Array(high)}

    kernel = SimpleCLFunction.from_string('''
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
    ''', dependencies=[Rand123()])

    return _generate_samples(kernel, nmr_distributions, nmr_samples, ctype, kernel_data, seed=seed)


def normal(nmr_distributions, nmr_samples, mean=0, std=1, ctype='float', seed=None):
    """Draw random samples from the Gaussian distribution.

    Args:
        nmr_distributions (int): the number of unique continuous_distributions to create
        nmr_samples (int): The number of samples to draw
        mean (float or ndarray): The mean of the distribution
        std (float or ndarray): The standard deviation or the distribution
        ctype (str): the C type of the output samples
        seed (float): the seed for the RNG

    Returns:
        ndarray: A two dimensional numpy array as (nmr_distributions, nmr_samples).
    """
    if is_scalar(mean):
        mean = np.ones((nmr_distributions, 1)) * mean
    if is_scalar(std):
        std = np.ones((nmr_distributions, 1)) * std

    kernel_data = {'mean': Array(mean),
                   'std': Array(std)}

    kernel = SimpleCLFunction.from_string('''
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
    ''', dependencies=[Rand123()])

    return _generate_samples(kernel, nmr_distributions, nmr_samples, ctype, kernel_data, seed=seed)


def _generate_samples(cl_function, nmr_distributions, nmr_samples, ctype, kernel_data, seed=None):
    np.random.seed(seed)
    rng_state = np.random.uniform(low=np.iinfo(np.uint32).min, high=np.iinfo(np.uint32).max + 1,
                                  size=(nmr_distributions, 6)).astype(np.uint32)

    kernel_data.update({'samples': Zeros((nmr_distributions, nmr_samples), ctype),
                        '_rng_state': Array(rng_state, 'uint')})
    cl_function.evaluate({'data': kernel_data}, nmr_instances=nmr_distributions)
    return kernel_data['samples'].get_data()
