import os
from random import Random

import numpy as np
from pkg_resources import resource_filename

__author__ = 'Robbert Harms'
__date__ = "2016-12-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Random123StartingPoint(object):

    def get_key(self):
        """Gets the key used as starting point for the Random123 generator.

        This should either return 0 or 2 keys depending on the desired precision.

        Returns:
            ndarray: 0 or 2 elements of dtype uint32.
        """

    def get_counter(self):
        """Gets the counter used as starting point for the Random123 generator.

        Returns:
            ndarray: 4 elements of dtype uint32.
        """


class StartingPointFromSeed(Random123StartingPoint):

    def __init__(self, seed, key_length=None):
        """Generates the key and counter from the given seed.

        Args:
            seed (int): the seed used to generate a starting point for the Random123 RNG.
            key_length (int): the length of the key, either 0 or 2 depending on the desired precision.
        """
        self._seed = seed
        self._key_length = key_length or 2
        if self._key_length not in (0, 2):
            raise ValueError('The key length should be either 0 or 2, {} given.'.format(key_length))

    def get_key(self):
        rng = Random(self._seed)
        dtype_info = np.iinfo(np.uint32)

        return np.array(list(rng.randrange(dtype_info.min, dtype_info.max + 1) for _ in range(self._key_length)),
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
    src += open(os.path.abspath(
        resource_filename('mot', 'data/opencl/random123/{}.h'.format(generator)), ), 'r').read()
    src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random.h'.format(generator)), ), 'r').read()
    src += (open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/rand123.cl'), ), 'r').read() % {
       'GENERATOR_FUNCTION': (generator + '4x32')
    })

    return src
