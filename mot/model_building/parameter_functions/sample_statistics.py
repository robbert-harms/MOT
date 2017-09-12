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
        from mot.cl_routines.mapping.circular_gaussian_fit import CircularGaussianFit
        mean, std = CircularGaussianFit().calculate(samples, high=self.max_angle, low=self.min_angle)
        return SamplingStatisticsContainer(mean, {'std': std})


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
        from mot.cl_routines.mapping.truncated_gaussian_fit import TruncatedGaussianFit as fitter
        mean, std = fitter().calculate(samples, high=self._high, low=self._low)
        return SamplingStatisticsContainer(mean, {'std': std})


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
