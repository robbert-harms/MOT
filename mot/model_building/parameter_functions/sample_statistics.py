import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2014-10-23"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ParameterSampleStatistics(object):

    def get_statistics(self, samples):
        """Get the statistics for this parameter.

        Args:
            samples (ndarray): The 2d matrix (v, s) with for v voxels, s samples.

        Returns:
            SamplingStatistics: an object containing the sampling statistics
        """
        raise NotImplementedError()


class GaussianPSS(ParameterSampleStatistics):
    """Calculates the mean and the standard deviation of the given samples.

    The standard deviation is calculated with a degree of freedom of one, meaning we are returning the unbiased
    estimator.
    """

    def get_statistics(self, samples):
        return SamplingStatisticsContainer(np.mean(samples, axis=1), {'std': np.std(samples, axis=1, ddof=1)})


class CircularGaussianPSS(ParameterSampleStatistics):

    def __init__(self):
        """Compute the circular mean for some samples.

        This assumes angles between 0 and pi.
        """
        super(CircularGaussianPSS, self).__init__()

    def get_statistics(self, samples):
        mean, std = CircularGaussianPSS.circmean_circstd(np.mod(samples, np.pi))
        return SamplingStatisticsContainer(mean, {'std': std})

    @staticmethod
    def circmean_circstd(samples):
        """Compute the circular mean for samples in a range.

        Copied and modified from scipy.stats.circmean and scipy.stats.circstd. Merged the two functions and
        hardcoded the limits between 0 and pi.

        Args:
            samples (array_like): Input array.

        Returns:
            tuple: (float, float) Circular mean and circular std
        """
        complex_coordinate_means = np.mean(np.exp(1j * 2 * samples), axis=1)

        R = abs(complex_coordinate_means)
        R[R >= 1] = 1 - np.finfo(np.float).eps
        stds = 1/2. * np.sqrt(-2 * np.log(R))

        res = np.angle(complex_coordinate_means)
        mask = res < 0
        if mask.ndim > 0:
            res[mask] += 2 * np.pi
        elif mask:
            res += 2 * np.pi

        return res/2., stds


class SamplingStatistics(object):

    def get_point_estimate(self):
        """Get the map that would represent the point estimate of these samples.

        This map is used for comparison with the point estimates obtained from optimization and typically corresponds
        to the mean of the distribution.

        Returns:
            ndarray: The point estimate for every voxel.
        """
        raise NotImplementedError()

    def get_additional_statistics(self):
        """Get additional statistics about the parameter distribution.

        This normally returns only a dictionary with a standard deviation map, but it can return more statistics
        if desired.

        Returns:
            dict: dictionary with additional statistics. Example: ``{'std': ...}``
        """
        raise NotImplementedError()


class SamplingStatisticsContainer(SamplingStatistics):

    def __init__(self, point_estimate, additional_maps):
        """Simple container for storing the point estimate and the other maps.

        Args:
            point_estimate (ndarray): the array with the point estimates
            additional_maps (dict): the additional maps
        """
        self._point_estimate = point_estimate
        self._additional_maps = additional_maps

    def get_point_estimate(self):
        return self._point_estimate

    def get_additional_statistics(self):
        return self._additional_maps
