"""This module contains some diagnostic functions to diagnose the performance of MCMC sampling.

The two most important functions are :func:`multivariate_ess` and :func:`univariate_ess` to calculate the effective
sample size of your samples.
"""
import os
from collections import Mapping
import multiprocessing
import itertools
import numpy as np
from numpy.linalg import det
from scipy.special import gammaln
from scipy.stats import chi2


__author__ = 'Robbert Harms'
__date__ = "2017-03-07"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def multivariate_ess(samples, batch_size_generator=None):
    r"""Estimate the multivariate Effective Sample Size for the samples of every problem.

    This essentially applies :func:`estimate_multivariate_ess` to every problem.

    Args:
        samples (ndarray, dict or generator): either an matrix of shape (d, p, n) with d problems, p parameters and
            n samples, or a dictionary with for every parameter a matrix with shape (d, n) or, finally,
            a generator function that yields sample arrays of shape (p, n).
        batch_size_generator (MultiVariateESSBatchSizeGenerator): the batch size generator, tells us how many
            batches and of which size we use in estimating the minimum ESS.

    Returns:
        ndarray: the multivariate ESS per problem
    """
    samples_generator = _get_sample_generator(samples)

    if os.name == 'nt': # In Windows there is no fork.
        return np.array(list(map(_MultivariateESSMultiProcessing(batch_size_generator),
                                 samples_generator())))

    try:
        p = multiprocessing.Pool()
        return_data = np.array(list(p.imap(_MultivariateESSMultiProcessing(batch_size_generator),
                                           samples_generator())))
        p.close()
        p.join()
        return return_data

    except OSError:
        return np.array(list(map(_MultivariateESSMultiProcessing(batch_size_generator),
                                 samples_generator())))


class _MultivariateESSMultiProcessing(object):

    def __init__(self, batch_size_generator):
        """Used in the function :func:`multivariate_ess` to estimate the multivariate ESS using multiprocessing."""
        self._batch_size_generator = batch_size_generator

    def __call__(self, samples):
        return estimate_multivariate_ess(samples, batch_size_generator=self._batch_size_generator)


def univariate_ess(samples, method='standard_error', **kwargs):
    r"""Estimate the univariate Effective Sample Size for the samples of every problem.

    This essentially applies the chosen univariate ESS method on every problem.

    Args:
        samples (ndarray, dict or generator): either an matrix of shape (d, p, n) with d problems, p parameters and
            n samples, or a dictionary with for every parameter a matrix with shape (d, n) or, finally,
            a generator function that yields sample arrays of shape (p, n).
        method (str): one of 'autocorrelation' or 'standard_error' defaults to 'standard_error'.
            If 'autocorrelation' is chosen we apply the function: :func:`estimate_univariate_ess_autocorrelation`,
            if 'standard_error` is choosen we apply the function: :func:`estimate_univariate_ess_standard_error`.
        **kwargs: passed to the chosen compute method

    Returns:
        ndarray: a matrix of size (d, p) with for every problem and every parameter an ESS.
    """
    samples_generator = _get_sample_generator(samples)

    if os.name == 'nt':  # In Windows there is no fork.
        return np.array(list(map(_UnivariateESSMultiProcessing(method, **kwargs),
                                 samples_generator())))

    p = multiprocessing.Pool()
    return_data = np.array(list(p.imap(_UnivariateESSMultiProcessing(method, **kwargs),
                                       samples_generator())))
    p.close()
    p.join()
    return return_data


class _UnivariateESSMultiProcessing(object):

    def __init__(self, method, **kwargs):
        """Used in the function :func:`univariate_ess` to estimate the univariate ESS using multiprocessing."""
        self._method = method
        self._kwargs = kwargs

    def __call__(self, samples):
        if self._method == 'autocorrelation':
            compute_func = estimate_univariate_ess_autocorrelation
        else:
            compute_func = estimate_univariate_ess_standard_error

        result = np.zeros(samples.shape[0])
        for param_ind in range(samples.shape[0]):
            result[param_ind] = compute_func(samples[param_ind], **self._kwargs)

        return result


def _get_sample_generator(samples):
    """Get a sample generator from the given polymorphic input.

    Args:
        samples (ndarray, dict or generator): either an matrix of shape (d, p, n) with d problems, p parameters and
            n samples, or a dictionary with for every parameter a matrix with shape (d, n) or, finally,
            a generator function that yields sample arrays of shape (p, n).

    Returns:
        generator: a generator that yields a matrix of size (p, n) for every problem in the input.
    """
    if isinstance(samples, Mapping):
        def samples_generator():
            for ind in range(samples[list(samples.keys())[0]].shape[0]):
                yield np.array([samples[s][ind, :] for s in sorted(samples)])
    elif isinstance(samples, np.ndarray):
        def samples_generator():
            for ind in range(samples.shape[0]):
                yield samples[ind]
    else:
        samples_generator = samples
    return samples_generator


def get_auto_correlation(chain, lag):
    r"""Estimates the auto correlation for the given chain (1d vector) with the given lag.

    Given a lag :math:`k`, the auto correlation coefficient :math:`\rho_{k}` is estimated as:

    .. math::

        \hat{\rho}_{k} = \frac{E[(X_{t} - \mu)(X_{t + k} - \mu)]}{\sigma^{2}}

    Please note that this equation only works for lags :math:`k < n` where :math:`n` is the number of samples in
    the chain.

    Args:
        chain (ndarray): the vector with the samples
        lag (int): the lag to use in the autocorrelation computation

    Returns:
        float: the autocorrelation with the given lag
    """
    normalized_chain = chain - np.mean(chain, dtype=np.float64)
    lagged_mean = np.mean(normalized_chain[:len(chain) - lag] * normalized_chain[lag:], dtype=np.float64)
    return lagged_mean / np.var(chain, dtype=np.float64)


def get_auto_correlation_time(chain, max_lag=None):
    r"""Compute the auto correlation time up to the given lag for the given chain (1d vector).

    This will halt when the maximum lag :math:`m` is reached or when the sum of two consecutive lags for any
    odd lag is lower or equal to zero.

    The auto correlation sum is estimated as:

    .. math::

        \tau = 1 + 2 * \sum_{k=1}^{m}{\rho_{k}}


    Where :math:`\rho_{k}` is estimated as:

    .. math::

        \hat{\rho}_{k} = \frac{E[(X_{t} - \mu)(X_{t + k} - \mu)]}{\sigma^{2}}

    Args:
        chain (ndarray): the vector with the samples
        max_lag (int): the maximum lag to use in the autocorrelation computation. If not given we use:
            :math:`min(n/3, 1000)`.
    """
    max_lag = max_lag or min(len(chain) // 3, 1000)

    normalized_chain = chain - np.mean(chain, dtype=np.float64)

    previous_accoeff = 0
    auto_corr_sum = 0

    for lag in range(1, max_lag):
        auto_correlation_coeff = np.mean(normalized_chain[:len(chain) - lag] * normalized_chain[lag:], dtype=np.float64)

        if lag % 2 == 0:
            if previous_accoeff + auto_correlation_coeff <= 0:
                break

        auto_corr_sum += auto_correlation_coeff
        previous_accoeff = auto_correlation_coeff

    return auto_corr_sum / np.var(chain, dtype=np.float64)


def estimate_univariate_ess_autocorrelation(chain, max_lag=None):
    r"""Estimate effective sample size (ESS) using the autocorrelation of the chain.

    The ESS is an estimate of the size of an iid sample with the same variance as the current sample.
    This function implements the ESS as described in Kass et al. (1998) and Robert and Casella (2004; p. 500):

    .. math::

        ESS(X) = \frac{n}{\tau} = \frac{n}{1 + 2 * \sum_{k=1}^{m}{\rho_{k}}}

    where :math:`\rho_{k}` is estimated as:

    .. math::

        \hat{\rho}_{k} = \frac{E[(X_{t} - \mu)(X_{t + k} - \mu)]}{\sigma^{2}}

    References:
        * Kass, R. E., Carlin, B. P., Gelman, A., and Neal, R. (1998)
            Markov chain Monte Carlo in practice: A roundtable discussion. The American Statistician, 52, 93--100.
        * Robert, C. P. and Casella, G. (2004) Monte Carlo Statistical Methods. New York: Springer.
        * Geyer, C. J. (1992) Practical Markov chain Monte Carlo. Statistical Science, 7, 473--483.

    Args:
        chain (ndarray): the chain for which to calculate the ESS, assumes a vector of length ``n`` samples
        max_lag (int): the maximum lag used in the variance calculations. If not given defaults to
            :math:`min(n/3, 1000)`.

    Returns:
        float: the estimated ESS
    """
    return len(chain) / (1 + 2 * get_auto_correlation_time(chain, max_lag))


def estimate_univariate_ess_standard_error(chain, batch_size_generator=None, compute_method=None):
    r"""Compute the univariate ESS using the standard error method.

    This computes the ESS using:

    .. math::

        ESS(X) = n * \frac{\lambda^{2}}{\sigma^{2}}

    Where :math:`\lambda` is the variance of the chain and :math:`\sigma` is estimated using the monte carlo
    standard error (which in turn is by default estimated using a batch means estimator).

    Args:
        chain (ndarray): the Markov chain
        batch_size_generator (UniVariateESSBatchSizeGenerator): the method that generates that batch sizes
            we will use. Per default it uses the :class:`SquareRootSingleBatch` method.
        compute_method (ComputeMonteCarloStandardError): the method used to compute the standard error.
            By default we will use the :class:`BatchMeansMCSE` method

    Returns:
        float: the estimated ESS
    """
    sigma = (monte_carlo_standard_error(chain, batch_size_generator=batch_size_generator,
                                        compute_method=compute_method) ** 2 * len(chain))
    lambda_ = np.var(chain, dtype=np.float64)
    return len(chain) * (lambda_ / sigma)


def minimum_multivariate_ess(nmr_params, alpha=0.05, epsilon=0.05):
    r"""Calculate the minimum multivariate Effective Sample Size you will need to obtain the desired precision.

    This implements the inequality from Vats et al. (2016):

    .. math::

        \widehat{ESS} \geq \frac{2^{2/p}\pi}{(p\Gamma(p/2))^{2/p}} \frac{\chi^{2}_{1-\alpha,p}}{\epsilon^{2}}

    Where :math:`p` is the number of free parameters.

    Args:
        nmr_params (int): the number of free parameters in the model
        alpha (float): the level of confidence of the confidence region. For example, an alpha of 0.05 means
            that we want to be in a 95% confidence region.
        epsilon (float): the level of precision in our multivariate ESS estimate.
            An epsilon of 0.05 means that we expect that the Monte Carlo error is 5% of the uncertainty in
            the target distribution.

    Returns:
        float: the minimum multivariate Effective Sample Size that one should aim for in MCMC sampling to
            obtain the desired confidence region with the desired precision.

    References:
        Vats D, Flegal J, Jones G (2016). Multivariate Output Analysis for Markov Chain Monte Carlo.
        arXiv:1512.07713v2 [math.ST]
    """
    tmp = 2.0 / nmr_params
    log_min_ess = tmp * np.log(2) + np.log(np.pi) - tmp * (np.log(nmr_params) + gammaln(nmr_params / 2)) \
                  + np.log(chi2.ppf(1 - alpha, nmr_params)) - 2 * np.log(epsilon)
    return int(round(np.exp(log_min_ess)))


def multivariate_ess_precision(nmr_params, multi_variate_ess, alpha=0.05):
    r"""Calculate the precision given your multivariate Effective Sample Size.

    Given that you obtained :math:`ESS` multivariate effective samples in your estimate you can calculate the
    precision with which you approximated your desired confidence region.

    This implements the inequality from Vats et al. (2016), slightly restructured to give :math:`\epsilon` back instead
    of the minimum ESS.

    .. math::

         \epsilon = \sqrt{\frac{2^{2/p}\pi}{(p\Gamma(p/2))^{2/p}} \frac{\chi^{2}_{1-\alpha,p}}{\widehat{ESS}}}

    Where :math:`p` is the number of free parameters and ESS is the multivariate ESS from your samples.

    Args:
        nmr_params (int): the number of free parameters in the model
        multi_variate_ess (int): the number of iid samples you obtained in your sampling results.
        alpha (float): the level of confidence of the confidence region. For example, an alpha of 0.05 means
            that we want to be in a 95% confidence region.

    Returns:
        float: the minimum multivariate Effective Sample Size that one should aim for in MCMC sampling to
            obtain the desired confidence region with the desired precision.

    References:
        Vats D, Flegal J, Jones G (2016). Multivariate Output Analysis for Markov Chain Monte Carlo.
        arXiv:1512.07713v2 [math.ST]
    """
    tmp = 2.0 / nmr_params
    log_min_ess = tmp * np.log(2) + np.log(np.pi) - tmp * (np.log(nmr_params) + gammaln(nmr_params / 2)) \
                  + np.log(chi2.ppf(1 - alpha, nmr_params)) - np.log(multi_variate_ess)
    return np.sqrt(np.exp(log_min_ess))


def estimate_multivariate_ess_sigma(samples, batch_size):
    r"""Calculates the Sigma matrix which is part of the multivariate ESS calculation.

    This implementation is based on the Matlab implementation found at: https://github.com/lacerbi/multiESS

    The Sigma matrix is defined as:

    .. math::

        \Sigma = \Lambda + 2 * \sum_{k=1}^{\infty}{Cov(Y_{1}, Y_{1+k})}

    Where :math:`Y` are our samples and :math:`\Lambda` is the covariance matrix of the samples.

    This implementation computes the :math:`\Sigma` matrix using a Batch Mean estimator using the given batch size.
    The batch size has to be :math:`1 \le b_n \le n` and a typical value is either :math:`\lfloor n^{1/2} \rfloor`
    for slow mixing chains or :math:`\lfloor n^{1/3} \rfloor` for reasonable mixing chains.

    If the length of the chain is longer than the sum of the length of all the batches, this implementation
    calculates :math:`\Sigma` for every offset and returns the average of those offsets.

    Args:
        samples (ndarray): the samples for which we compute the sigma matrix. Expects an (p, n) array with
            p the number of parameters and n the sample size
        batch_size (int): the batch size used in the approximation of the correlation covariance

    Returns:
        ndarray: an pxp array with p the number of parameters in the samples.

    References:
        Vats D, Flegal J, Jones G (2016). Multivariate Output Analysis for Markov Chain Monte Carlo.
        arXiv:1512.07713v2 [math.ST]
    """
    sample_means = np.mean(samples, axis=1, dtype=np.float64)
    nmr_params, chain_length = samples.shape

    nmr_batches = int(np.floor(chain_length / batch_size))
    sigma = np.zeros((nmr_params, nmr_params))

    nmr_offsets = chain_length - nmr_batches * batch_size + 1

    for offset in range(nmr_offsets):
        batches = np.reshape(samples[:, np.array(offset + np.arange(0, nmr_batches * batch_size), dtype=np.int)].T,
                             [batch_size, nmr_batches, nmr_params], order='F')

        batch_means = np.squeeze(np.mean(batches, axis=0, dtype=np.float64))

        Z = batch_means - sample_means

        for x, y in itertools.product(range(nmr_params), range(nmr_params)):
            sigma[x, y] += np.sum(Z[:, x] * Z[:, y])

    return sigma * batch_size / (nmr_batches - 1) / nmr_offsets


def estimate_multivariate_ess(samples, batch_size_generator=None, full_output=False):
    r"""Compute the multivariate Effective Sample Size of your (single instance set of) samples.

    This multivariate ESS is defined in Vats et al. (2016) and is given by:

    .. math::

        ESS = n \bigg(\frac{|\Lambda|}{|\Sigma|}\bigg)^{1/p}

    Where :math:`n` is the number of samples, :math:`p` the number of parameters, :math:`\Lambda` is the covariance
    matrix of the parameters and :math:`\Sigma` captures the covariance structure in the target together with
    the covariance due to correlated samples. :math:`\Sigma` is estimated using
    :func:`estimate_multivariate_ess_sigma`.

    In the case of NaN in any part of the computation the ESS is set to 0.

    To compute the multivariate ESS for multiple problems, please use :func:`multivariate_ess`.

    Args:
        samples (ndarray): an pxn matrix with for p parameters and n samples.
        batch_size_generator (MultiVariateESSBatchSizeGenerator): the batch size generator, tells us how many
            batches and of which size we use for estimating the minimum ESS. Defaults to :class:`SquareRootSingleBatch`
        full_output (boolean): set to True to return the estimated :math:`\Sigma` and the optimal batch size.

    Returns:
        float or tuple: when full_output is set to True we return a tuple with the estimated multivariate ESS,
            the estimated :math:`\Sigma` matrix and the optimal batch size. When full_output is False (the default)
            we only return the ESS.

    References:
        Vats D, Flegal J, Jones G (2016). Multivariate Output Analysis for Markov Chain Monte Carlo.
        arXiv:1512.07713v2 [math.ST]
    """
    batch_size_generator = batch_size_generator or SquareRootSingleBatch()

    batch_sizes = batch_size_generator.get_multivariate_ess_batch_sizes(*samples.shape)

    nmr_params, chain_length = samples.shape
    nmr_batches = len(batch_sizes)

    det_lambda = det(np.cov(samples))

    ess_estimates = np.zeros(nmr_batches)
    sigma_estimates = np.zeros((nmr_params, nmr_params, nmr_batches))

    for i in range(0, nmr_batches):
        sigma = estimate_multivariate_ess_sigma(samples, int(batch_sizes[i]))
        ess = chain_length * (det_lambda**(1.0 / nmr_params) / det(sigma)**(1.0 / nmr_params))

        ess_estimates[i] = ess
        sigma_estimates[..., i] = sigma

    ess_estimates = np.nan_to_num(ess_estimates)

    if nmr_batches > 1:
        idx = np.argmin(ess_estimates)
    else:
        idx = 0

    if full_output:
        return ess_estimates[idx], sigma_estimates[..., idx], batch_sizes[idx]
    return ess_estimates[idx]


def monte_carlo_standard_error(chain, batch_size_generator=None, compute_method=None):
    """Compute Monte Carlo standard errors for the expectations

    This is a convenience function that calls the compute method for each batch size and returns the lowest ESS
    over the used batch sizes.

    Args:
        chain (ndarray): the Markov chain
        batch_size_generator (UniVariateESSBatchSizeGenerator): the method that generates that batch sizes
            we will use. Per default it uses the :class:`SquareRootSingleBatch` method.
        compute_method (ComputeMonteCarloStandardError): the method used to compute the standard error.
            By default we will use the :class:`BatchMeansMCSE` method
    """
    batch_size_generator = batch_size_generator or SquareRootSingleBatch()
    compute_method = compute_method or BatchMeansMCSE()

    batch_sizes = batch_size_generator.get_univariate_ess_batch_sizes(len(chain))

    return np.min(list(compute_method.compute_standard_error(chain, b) for b in batch_sizes))


class MultiVariateESSBatchSizeGenerator(object):
    """Objects of this class are used as input to the multivariate ESS function.

    The multivariate ESS function needs to have at least one batch size to use during the computations. More batch
    sizes are also possible and the batch size with the lowest ESS is then preferred.
    Objects of this class implement the logic behind choosing batch sizes.
    """

    def get_multivariate_ess_batch_sizes(self, nmr_params, chain_length):
        r"""Get the batch sizes to use for the calculation of the Effective Sample Size (ESS).

        This should return a list of batch sizes that the ESS calculation will use to determine :math:`\Sigma`

        Args:
            nmr_params (int): the number of parameters in the samples
            chain_length (int): the length of the chain

        Returns:
            list: the batches of the given sizes we will test in the ESS calculations
        """


class UniVariateESSBatchSizeGenerator(object):
    """Objects of this class are used as input to the univariate ESS function that uses the batch means.

    The univariate batch means ESS function needs to have at least one batch size to use during the computations.
    More batch sizes are also possible and the batch size with the lowest ESS is then preferred.
    Objects of this class implement the logic behind choosing batch sizes.
    """

    def get_univariate_ess_batch_sizes(self, chain_length):
        r"""Get the batch sizes to use for the calculation of the univariate Effective Sample Size (ESS).

        This should return a list of batch sizes that the ESS calculation will use to determine :math:`\sigma`

        Args:
            chain_length (int): the length of the chain

        Returns:
            list: the batches of the given sizes we will test in the ESS calculations
        """


class SquareRootSingleBatch(MultiVariateESSBatchSizeGenerator, UniVariateESSBatchSizeGenerator):
    r"""Returns :math:`\sqrt(n)`."""

    def get_multivariate_ess_batch_sizes(self, nmr_params, chain_length):
        return [np.floor(chain_length**(1/2.0))]

    def get_univariate_ess_batch_sizes(self, chain_length):
        return [np.floor(chain_length ** (1 / 2.0))]


class CubeRootSingleBatch(MultiVariateESSBatchSizeGenerator, UniVariateESSBatchSizeGenerator):
    r"""Returns :math:`n^{1/3}`."""

    def get_multivariate_ess_batch_sizes(self, nmr_params, chain_length):
        return [np.floor(chain_length**(1/3.0))]

    def get_univariate_ess_batch_sizes(self, chain_length):
        return [np.floor(chain_length ** (1 / 3.0))]


class LinearSpacedBatchSizes(MultiVariateESSBatchSizeGenerator):

    def __init__(self, nmr_batches=200):
        r"""Returns a number of batch sizes from which the ESS algorithm will select the one with the lowest ESS.

        This is a conservative choice since the lowest ESS of all batch sizes is chosen.

        The batch sizes are generated as linearly spaced values in:

        .. math::

            \Big[ n^{1/4}, max(\lfloor x/max(20,p) \rfloor, \lfloor \sqrt{n} \rfloor) \Big]

        where :math:`n` is the chain length and :math:`p` is the number of parameters.

        Args:
            nmr_batches (int): the number of linearly spaced batches we will generate.
        """
        self._nmr_batches = nmr_batches

    def get_multivariate_ess_batch_sizes(self, nmr_params, chain_length):
        b_min = np.floor(chain_length**(1 / 4.0))
        b_max = np.max((np.floor(chain_length / np.max((nmr_params, 20))),
                        np.floor(chain_length**(1 / 2.0))))
        return list(np.unique(np.round(np.exp(np.linspace(np.log(b_min), np.log(b_max), self._nmr_batches)))))


class ComputeMonteCarloStandardError(object):
    """Method to compute the Monte Carlo Standard error."""

    def compute_standard_error(self, chain, batch_size):
        """Compute the standard error of the given chain and the given batch size.

        Args:
            chain (ndarray): the chain for which to compute the SE
            batch_size (int): batch size or window size to use in the computations

        Returns:
            float: the Monte Carlo Standard Error
        """
        raise NotImplementedError()


class BatchMeansMCSE(ComputeMonteCarloStandardError):
    """Computes the Monte Carlo Standard Error using simple batch means."""

    def compute_standard_error(self, chain, batch_size):
        nmr_batches = int(np.floor(len(chain) / batch_size))

        batch_means = np.zeros(nmr_batches)

        for batch_index in range(nmr_batches):
            batch_means[batch_index] = np.mean(
                chain[int(batch_index * batch_size):int((batch_index + 1) * batch_size)], dtype=np.float64)

        var_hat = batch_size * sum((batch_means - np.mean(chain, dtype=np.float64))**2) / (nmr_batches - 1)
        return np.sqrt(var_hat / len(chain))


class OverlappingBatchMeansMCSE(ComputeMonteCarloStandardError):
    """Computes the Monte Carlo Standard Error using overlapping batch means."""

    def compute_standard_error(self, chain, batch_size):
        nmr_batches = int(len(chain) - batch_size + 1)

        batch_means = np.zeros(nmr_batches)

        for batch_index in range(nmr_batches):
            batch_means[batch_index] = np.mean(chain[int(batch_index):int(batch_index + batch_size)], dtype=np.float64)

        var_hat = (len(chain) * batch_size
                   * sum((batch_means - np.mean(chain, dtype=np.float64))**2)) / (nmr_batches - 1) / nmr_batches
        return np.sqrt(var_hat / len(chain))
