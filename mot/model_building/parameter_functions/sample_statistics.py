import multiprocessing
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import os

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

        This will do a maximum likelihood estimation of the truncated gaussian on the given data.

        Args:
            low (float): the lower bound of the truncated gaussian
            high (float): the upper bound of the truncated gaussian
        """
        self._low = low
        self._high = high

    def get_statistics(self, samples):
        fitter = _TruncatedNormalFitter(self._low, self._high)

        def samples_generator():
            for ind in range(samples.shape[0]):
                yield samples[ind]

        if os.name == 'nt':  # In Windows there is no fork.
            results = np.array(list(map(fitter, samples_generator())), dtype=samples.dtype)
        else:
            try:
                p = multiprocessing.Pool()
                results = np.array(list(p.imap(fitter, samples_generator())), dtype=samples.dtype)
                p.close()
                p.join()
            except OSError:
                results = np.array(list(map(fitter, samples_generator())), dtype=samples.dtype)

        return SamplingStatisticsContainer(results[:, 0], {'std': results[:, 1]})

class _TruncatedNormalFitter(object):

    def __init__(self, low, high):
        """Fit the mean and std of the truncated normal to the given samples.

        This is in a separate class to use the python multiprocessing library.
        """
        self._low = low
        self._high = high

    def __call__(self, samples):
        result = minimize(_TruncatedNormalFitter.truncated_normal_log_likelihood,
                          np.array([np.mean(samples), np.std(samples)]),
                          args=(self._low, self._high, samples,),
                          method='TNC',
                          jac=_TruncatedNormalFitter.truncated_normal_ll_gradient,
                          bounds=[(self._low, self._high), (0, None)],
                          options=dict(maxiter=20))

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
        return [_TruncatedNormalFitter.partial_derivative_mu(params[0], params[1], low, high, data),
                _TruncatedNormalFitter.partial_derivative_sigma(params[0], params[1], low, high, data)]

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
