import numpy as np
from scipy.stats import truncnorm

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


class GaussianFit(ParameterSampleStatistics):
    """Calculates the mean and the standard deviation of the given samples.

    The standard deviation is calculated with a degree of freedom of one, meaning we are returning the unbiased
    estimator.
    """

    def get_statistics(self, samples):
        return SamplingStatisticsContainer(np.mean(samples, axis=1), {'std': np.std(samples, axis=1, ddof=1)})


class CircularGaussianFit(ParameterSampleStatistics):

    def __init__(self, max_angle=np.pi, min_angle=0):
        """Compute the circular mean for samples in a range

        The minimum angle is set to 0, the maximum angle can be given.

        Args:
            max_angle (float): The maximum angle used in the calculations
            min_angle (float): The minimum wrapped angle
        """
        super(CircularGaussianFit, self).__init__()
        self.max_angle = max_angle
        self.min_angle = min_angle

    def get_statistics(self, samples):
        mod = np.mod(samples, self.max_angle)

        mean = CircularGaussianFit.circmean(mod, high=self.max_angle, low=self.min_angle, axis=1)
        additional_maps = {'std': CircularGaussianFit.circstd(mod, high=self.max_angle, low=self.min_angle, axis=1)}

        return SamplingStatisticsContainer(mean, additional_maps)

    @staticmethod
    def circmean(samples, high=2*np.pi, low=0, axis=None):
        """Compute the circular mean for samples in a range.
        Taken from scipy.stats

        Args:
            samples (array_like): Input array.
            high (float or int): High boundary for circular mean range.  Default is ``2*pi``.
            low (float or int): Low boundary for circular mean range.  Default is 0.
            axis (int, optional): Axis along which means are computed.
                The default is to compute the mean of the flattened array.

        Returns:
            float: Circular mean.
        """
        ang = (samples - low) * 2 * np.pi / (high - low)
        res = np.angle(np.mean(np.exp(1j * ang), axis=axis))
        mask = res < 0
        if mask.ndim > 0:
            res[mask] += 2 * np.pi
        elif mask:
            res += 2 * np.pi
        return res * (high - low) / 2.0 / np.pi + low

    @staticmethod
    def circstd(samples, high=2*np.pi, low=0, axis=None):
        """Compute the circular standard deviation for samples assumed to be in the range [low to high].

        Taken from scipy.stats, with a small change on the 4th line.

        This uses a definition of circular standard deviation that in the limit of
        small angles returns a number close to the 'linear' standard deviation.

        Args:
            samples (array_like): Input array.
            low (float or int): Low boundary for circular standard deviation range.  Default is 0.
            high (float or int): High boundary for circular standard deviation range. Default is ``2*pi``.
            axis (int): Axis along which standard deviations are computed.  The default is
                to compute the standard deviation of the flattened array.

        Returns:
            float: Circular standard deviation.
        """
        ang = (samples - low) * 2 * np.pi / (high - low)
        res = np.mean(np.exp(1j * ang), axis=axis)
        R = abs(res)
        R[R >= 1] = 1 - np.finfo(np.float).eps
        return ((high - low) / 2.0 / np.pi) * np.sqrt(-2 * np.log(R))


class CircularGaussianPIFit(ParameterSampleStatistics):
    """Compute the circular mean where the results are wrapped around [0, pi].

    This is exactly the same as using CircularGaussianFit from above with a maximum angle of pi. This special
    case is only faster to compute.

    This assumes angles between 0 and pi.
    """

    def get_statistics(self, samples):
        mean, std = self.circmean_circstd(np.mod(samples, np.pi))
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


class TruncatedGaussianFit(ParameterSampleStatistics):

    def __init__(self, low, high):
        """Fits a truncated gaussian distribution on the given samples.

        This may return mean values outside the bounds.

        Args:
            low (float): the lower bound of the truncated gaussian
            high (float): the upper bound of the truncated gaussian
        """
        self._low = low
        self._high = high

    def get_statistics(self, samples):
        means = np.zeros(samples.shape[0])
        stds = np.zeros(samples.shape[0])

        for ind in range(samples.shape[0]):
            mean, std = truncnorm.fit_loc_scale(samples[ind, :], self._low, self._high)
            means[ind] = np.clip(mean, self._low, self._high)
            stds[ind] = std

        return SamplingStatisticsContainer(means, {'std': stds})


class TruncatedGaussianFitClipped(TruncatedGaussianFit):

    def __init__(self, low, high):
        """Fits a truncated gaussian distribution on the given samples and clips the mean to the given bounds.

        Args:
            low (float): the lower bound of the truncated gaussian
            high (float): the upper bound of the truncated gaussian
        """
        super(TruncatedGaussianFitClipped, self).__init__(low, high)

    def get_statistics(self, samples):
        statistics = super(TruncatedGaussianFitClipped, self).get_statistics(samples)
        clipped_expected_value = np.clip(statistics.get_expected_value(), self._low, self._high)
        return SamplingStatisticsContainer(clipped_expected_value, statistics.get_additional_statistics())


class TruncatedGaussianFitModulus(TruncatedGaussianFit):

    def __init__(self, low, high, modulus):
        """Fits a truncated gaussian distribution on the given samples and applies the modulus on the mean.

        Args:
            low (float): the lower bound of the truncated gaussian
            high (float): the upper bound of the truncated gaussian
            modulus (float): wrap the mean around this modulus
        """
        super(TruncatedGaussianFitModulus, self).__init__(low, high)
        self._modulus = modulus

    def get_statistics(self, samples):
        statistics = super(TruncatedGaussianFitModulus, self).get_statistics(samples)
        modulus_expected_value = np.mod(statistics.get_expected_value(), self._modulus)
        return SamplingStatisticsContainer(modulus_expected_value, statistics.get_additional_statistics())


class SamplingStatistics(object):

    def get_expected_value(self):
        """Get the expected value (typically the mean) of the given dataset.

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

    def __init__(self, expected_value, additional_maps):
        """Simple container for storing the point estimate and the other maps.

        Args:
            expected_value (ndarray): the array with the expected value (mean)
            additional_maps (dict): the additional maps
        """
        self._expected_value = expected_value
        self._additional_maps = additional_maps

    def get_expected_value(self):
        return self._expected_value

    def get_additional_statistics(self):
        return self._additional_maps
