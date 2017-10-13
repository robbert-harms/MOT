import multiprocessing
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import os

from mot.utils import is_scalar

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
        return SimpleSamplingStatistics(np.mean(samples, axis=1), {'std': np.std(samples, axis=1, ddof=1)})

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
        from mot.cl_routines.mapping.circular_gaussian_fit import CircularGaussianFit
        mean, std = CircularGaussianFit().calculate(samples, high=self.high, low=self.low)
        return SimpleSamplingStatistics(mean, {'std': std})

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

        def item_generator():
            for ind in range(samples.shape[0]):
                if is_scalar(lower_bounds):
                    lower_bound = lower_bounds
                else:
                    lower_bound = lower_bounds[ind]

                if is_scalar(upper_bounds):
                    upper_bound = upper_bounds
                else:
                    upper_bound = upper_bounds[ind]

                yield (samples[ind] * self._scaling_factor,
                       lower_bound * self._scaling_factor,
                       upper_bound * self._scaling_factor)

        if os.name == 'nt':  # In Windows there is no fork.
            results = np.array(list(map(_TruncatedNormalFitter(), item_generator())), dtype=samples.dtype)
        else:
            try:
                p = multiprocessing.Pool()
                results = np.array(list(p.imap(_TruncatedNormalFitter(), item_generator())), dtype=samples.dtype)
                p.close()
                p.join()
            except OSError:
                results = np.array(list(map(_TruncatedNormalFitter(), item_generator())), dtype=samples.dtype)

        results /= self._scaling_factor
        return SimpleSamplingStatistics(results[:, 0], {'std': results[:, 1]})


class _TruncatedNormalFitter(object):

    def __call__(self, item):
        """Fit the mean and std of the truncated normal to the given samples.

        This is in a separate class to use the python multiprocessing library.
        """
        samples, lower_bound, upper_bound = item
        result = minimize(_TruncatedNormalFitter.truncated_normal_log_likelihood,
                          np.array([np.mean(samples), np.std(samples)]),
                          args=(lower_bound, upper_bound, samples),
                          method='TNC',
                          jac=_TruncatedNormalFitter.truncated_normal_ll_gradient,
                          bounds=[(lower_bound, upper_bound), (0, None)])
        return result.x

    @staticmethod
    def truncated_normal_log_likelihood(params, low, high, data):
        """Calculate the log likelihood of the truncated normal distribution.

        Args:
            params: tuple with (mean, std), the parameters under which we evaluate the model
            low (float): the lower truncation bound
            high (float): the upper truncation bound
            data (ndarray): the one dimension list of data points for which we want to calculate the likelihood

        Returns:
            float: the negative log likelihood of observing the given data under the given parameters.
                This is meant to be used in minimization routines.
        """
        mu = params[0]
        sigma = params[1]
        ll = np.sum(norm.logpdf(data, mu, sigma))
        ll -= len(data) * np.log((norm.cdf(high, mu, sigma) - norm.cdf(low, mu, sigma)))
        return -ll

    @staticmethod
    def truncated_normal_ll_gradient(params, low, high, data):
        """Return the gradient of the log likelihood of the truncated normal at the given position.

        Args:
            params: tuple with (mean, std), the parameters under which we evaluate the model
            low (float): the lower truncation bound
            high (float): the upper truncation bound
            data (ndarray): the one dimension list of data points for which we want to calculate the likelihood

        Returns:
            tuple: the gradient of the log likelihood given as a tuple with (mean, std)
        """
        return np.array([_TruncatedNormalFitter.partial_derivative_mu(params[0], params[1], low, high, data),
                         _TruncatedNormalFitter.partial_derivative_sigma(params[0], params[1], low, high, data)])

    @staticmethod
    def partial_derivative_mu(mu, sigma, low, high, data):
        """The partial derivative with respect to the mean.

        Args:
            mu (float): the mean of the truncated normal
            sigma (float): the std of the truncated normal
            low (float): the lower truncation bound
            high (float): the upper truncation bound
            data (ndarray): the one dimension list of data points for which we want to calculate the likelihood

        Returns:
            float: the partial derivative evaluated at the given point
        """
        pd_mu = np.sum(data - mu) / sigma ** 2
        pd_mu -= len(data) * ((norm.pdf(low, mu, sigma) - norm.pdf(high, mu, sigma))
                              / (norm.cdf(high, mu, sigma) - norm.cdf(low, mu, sigma)))
        return -pd_mu

    @staticmethod
    def partial_derivative_sigma(mu, sigma, low, high, data):
        """The partial derivative with respect to the standard deviation.

        Args:
            mu (float): the mean of the truncated normal
            sigma (float): the std of the truncated normal
            low (float): the lower truncation bound
            high (float): the upper truncation bound
            data (ndarray): the one dimension list of data points for which we want to calculate the likelihood

        Returns:
            float: the partial derivative evaluated at the given point
        """
        pd_sigma = np.sum(-(1 / sigma) + ((data - mu) ** 2 / (sigma ** 3)))
        pd_sigma -= len(data) * (((low - mu) * norm.pdf(low, mu, sigma) - (high - mu) * norm.pdf(high, mu, sigma))
                                 / (sigma * (norm.cdf(high, mu, sigma) - norm.cdf(low, mu, sigma))))
        return -pd_sigma


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


class SimpleSamplingStatistics(SamplingStatistics):

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
