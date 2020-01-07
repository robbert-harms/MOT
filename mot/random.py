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

    kernel_data = {'low': Array(low, as_scalar=True),
                   'high': Array(high, as_scalar=True),
                   'samples': Zeros((nmr_distributions, nmr_samples), ctype)}

    kernel = SimpleCLFunction.from_string('''
        void compute(double low, double high, ''' + ctype + '''* samples){
            for(uint i = 0; i < ''' + str(nmr_samples) + '''; i++){
                samples[i] = (''' + ctype + ''')(low + rand() * (high - low));
            }
        }
    ''')

    kernel.evaluate(kernel_data, nmr_distributions, enable_rng=True)
    return kernel_data['samples'].get_data()


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

    kernel_data = {'mean': Array(mean, as_scalar=True),
                   'std': Array(std, as_scalar=True),
                   'samples': Zeros((nmr_distributions, nmr_samples), ctype)}

    kernel = SimpleCLFunction.from_string('''
        void compute(double mean, double std, ''' + ctype + '''* samples){
            for(uint i = 0; i < ''' + str(nmr_samples) + '''; i++){
                samples[i] = (''' + ctype + ''')(mean + randn() * std);
            }
        }
    ''')

    kernel.evaluate(kernel_data, nmr_distributions, enable_rng=True)
    return kernel_data['samples'].get_data()
