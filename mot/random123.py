import os
from random import Random
import numpy as np
from pkg_resources import resource_filename

from mot.cl_routines.base import CLRoutine

__author__ = 'Robbert Harms'
__date__ = "2016-12-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Random123StartingPoint(CLRoutine):

    def get_key(self):
        """Gets the key used as starting point for the Random123 generator.

        Returns:
            ndarray: 2 elements of dtype uint32.
        """

    def get_counter(self):
        """Gets the counter used as starting point for the Random123 generator.

        Returns:
            ndarray: 4 elements of dtype uint32.
        """


class StartingPointFromSeed(Random123StartingPoint):

    def __init__(self, seed, **kwargs):
        """Generates the key and counter from the given seed.

        Args:
            seed (int): the seed used to generate a starting point for the Random123 RNG.
            key_length (int): the length of the key, either 0 or 2 depending on the desired precision.
        """
        super(StartingPointFromSeed, self).__init__(**kwargs)
        self._seed = seed

    def get_key(self):
        rng = Random(self._seed)
        dtype_info = np.iinfo(np.uint32)

        return np.array(list(rng.randrange(dtype_info.min, dtype_info.max + 1) for _ in range(2)),
                        dtype=np.uint32)

    def get_counter(self):
        rng = Random(self._seed + 1)
        dtype_info = np.iinfo(np.uint32)

        return np.array(list(rng.randrange(dtype_info.min, dtype_info.max + 1) for _ in range(4)), dtype=np.uint32)


class RandomStartingPoint(StartingPointFromSeed):

    def __init__(self, **kwargs):
        """Generates the key and counter randomly."""
        super(RandomStartingPoint, self).__init__(Random().randint(0, 2**31), **kwargs)


def get_random123_cl_code():
    """Get the source code needed for working with the Rand123 RNG.

    Returns:
        str: the CL code for the Rand123 RNG
    """
    generator = 'threefry'

    src = open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/openclfeatures.h'), ), 'r').read()
    src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/array.h'), ), 'r').read()
    src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/{}.h'.format(generator)), ), 'r').read()
    src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random.h'.format(generator)), ), 'r').read()
    src += (open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/rand123.cl'), ), 'r').read() % {
       'GENERATOR_NAME': (generator)
    })
    return src


def generate_uniform(nmr_samples, minimum=0, maximum=1, dtype=None, starting_point=None):
    """Draw random samples from the uniform distribution.

    Args:
        nmr_samples (int): The number of samples to draw
        minimum (double): The minimum value of the random numbers
        maximum (double): The minimum value of the random numbers
        dtype (np.dtype): the numpy datatype, either one of float32 (default) or float64.
        starting_point (Random123StartingPoint): the starting point for the RNG

    Returns:
        ndarray: A numpy array with nmr_samples random samples drawn from the uniform distribution.
    """
    from mot.cl_routines.generate_random import Random123GeneratorBase
    generator = Random123GeneratorBase(starting_point=starting_point)
    return generator.generate_uniform(nmr_samples, minimum=minimum, maximum=maximum, dtype=dtype)


def generate_gaussian(nmr_samples, mean=0, std=1, dtype=None, starting_point=None):
    """Draw random samples from the Gaussian distribution.

    Args:
        nmr_samples (int): The number of samples to draw
        mean (double): The mean of the distribution
        std (double): The standard deviation or the distribution
        dtype (np.dtype): the numpy datatype, either one of float32 (default) or float64.
        starting_point (Random123StartingPoint): the starting point for the RNG

    Returns:
        ndarray: A numpy array with nmr_samples random samples drawn from the Gaussian distribution.
    """
    from mot.cl_routines.generate_random import Random123GeneratorBase
    generator = Random123GeneratorBase(starting_point=starting_point)
    return generator.generate_gaussian(nmr_samples, mean=mean, std=std, dtype=dtype)
