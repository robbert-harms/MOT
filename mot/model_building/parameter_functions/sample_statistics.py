import numpy as np
from mot.statistics import fit_gaussian, fit_truncated_gaussian, fit_circular_gaussian

__author__ = 'Robbert Harms'
__date__ = "2014-10-23"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ParameterSampleStatistics(object):

    def get_statistics(self, samples, lower_bounds, upper_bounds):
        """Get the statistics for this parameter.

        Args:
            samples (ndarray): The 2d matrix (p, s) with for p problems, s samples.
            lower_bounds (ndarray or float): the lower bound(s) for this parameter. This is either a scalar with
                one lower bound for every problem, or a vector with a lower bound per problem.
            upper_bounds (ndarray or float): the upper bound(s) for this parameter. This is either a scalar with
                one upper bound for every problem, or a vector with a lower bound per problem.

        Returns:
            SamplingStatistics: an object containing the sampling statistics
        """
        raise NotImplementedError()

    def get_distance_from_expected(self, samples, expected_value):
        """Get the distance from the expected value according to this sampling statistic.

        This is used in the computation of the sample covariance matrix. For most implementations this is simply
        ``samples - mean``. For circular distributions this might be different.

        Args:
            samples (ndarray): The 2d matrix (p, s) with for p problems, s samples
            expected_value (ndarray): A 1d array with the expected values for each of the p problems.
                This should be computed using :meth:`get_statistics`.

        Returns:
            ndarray: a 2d array of the same size as the samples, containing the distances to the mean for
                the given parameter.
        """
        raise NotImplementedError()


class GaussianFit(ParameterSampleStatistics):
    """Calculates the mean and the standard deviation of the given samples.

    The standard deviation is calculated with a degree of freedom of one, meaning we are returning the unbiased
    estimator.
    """
    def get_statistics(self, samples, lower_bounds, upper_bounds):
        mean, std = fit_gaussian(samples, ddof=1)
        return SimpleSamplingStatistics(mean, std)

    def get_distance_from_expected(self, samples, expected_value):
        return samples - expected_value[:, None]


class CircularGaussianFit(ParameterSampleStatistics):

    def __init__(self, high=np.pi, low=0):
        """Compute the circular mean for samples in a range

        Args:
            high (float): The maximum wrap point
            low (float): The minimum wrap point
        """
        super(CircularGaussianFit, self).__init__()
        self.high = high
        self.low = low

    def get_statistics(self, samples, lower_bounds, upper_bounds):
        mean, std = fit_circular_gaussian(samples, high=self.high, low=self.low)
        return SimpleSamplingStatistics(mean, std)

    def get_distance_from_expected(self, samples, expected_value):
        distance_direct = samples - expected_value[:, None]

        distance_circular_chooser = expected_value[:, None] > samples
        distance_circular = \
            distance_circular_chooser * ((self.high - expected_value[:, None]) + (samples - self.low)) \
            - np.logical_not(distance_circular_chooser) * ((self.high - samples) + (expected_value[:, None] - self.low))

        use_circular = np.abs(distance_circular) < np.abs(distance_direct)
        return use_circular * distance_circular + np.logical_not(use_circular) * distance_direct


class TruncatedGaussianFit(ParameterSampleStatistics):

    def __init__(self, scaling_factor=1):
        """Fits a truncated gaussian distribution on the given samples.

        This will do a maximum likelihood estimation of the truncated gaussian on the given data where the
        truncation points are given by the lower and upper bounds.

        Args:
            scaling_factor (float): optionally scale the data with this factor before parameter estimation.
                This can improve accuracy when the data is in a very high or very low range.
        """
        self._scaling_factor = scaling_factor

    def get_distance_from_expected(self, samples, expected_value):
        return samples - expected_value[:, None]

    def get_statistics(self, samples, lower_bounds, upper_bounds):
        mean, std = fit_truncated_gaussian(samples*self._scaling_factor,
                                           lower_bounds*self._scaling_factor,
                                           upper_bounds* self._scaling_factor)
        return SimpleSamplingStatistics(mean / self._scaling_factor, std / self._scaling_factor)


class SamplingStatistics(object):

    @property
    def mean(self):
        """Get the mean of the calculated statistics.

        Returns:
            ndarray: The mean or point-estimate for every problem element.
        """
        raise NotImplementedError()

    @property
    def std(self):
        """Get the standard deviation of the calculated statistics.

        This should return the positive square root of the second central moment, the variance.

        Returns:
            ndarray: the standard deviation for every problem element.
        """
        raise NotImplementedError()

    def get_additional_statistics(self):
        """Get additional statistics about the parameter distribution.

        This should return a dictionary with additional elements to be stored for this distribution. For example,
        after fitting a Beta distribution this can return the found alpha and beta parameters.

        Returns:
            dict: dictionary with additional statistics. Example: ``{'alpha': <ndarray>}``
        """
        raise NotImplementedError()


class SimpleSamplingStatistics(SamplingStatistics):

    def __init__(self, mean, std, additional_maps=None):
        """Simple container for storing the point estimate and the other maps.

        Args:
            mean (ndarray): the array with the expected value (mean)
            std (ndarray): the array with the standard deviations
            additional_maps (dict): the additional maps
        """
        self._mean = mean
        self._std = std
        self._additional_statistics = additional_maps or {}

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def get_additional_statistics(self):
        return self._additional_statistics
