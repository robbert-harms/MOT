from collections import Mapping

import numpy as np
from numpy.linalg import det
from scipy.special._ufuncs import gammaln
from scipy.stats import chi2

__author__ = 'Robbert Harms'
__date__ = "2017-03-07"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


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
    normalized_chain = chain - np.mean(chain)
    return (np.mean(normalized_chain[:len(chain) - lag] * normalized_chain[lag:])) / np.var(chain)


def get_auto_correlation_time(chain, max_lag=None):
    r"""Compute the auto correlation time up to the given lag for the given chain (1d vector).

    This will break either when the maximum lag :math:`m` is reached or when the sum of two consecutive
    lags is lower or equal to zero.

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

    variance = np.var(chain)

    normalized_chain = chain - np.mean(chain)

    previous_arcoeff = variance
    auto_corr_sum = 0

    for lag in range(1, max_lag):
        auto_correlation_coeff = np.mean(normalized_chain[:len(chain) - lag] * normalized_chain[lag:])

        if lag % 2 == 0:
            if previous_arcoeff + auto_correlation_coeff <= 0:
                break

        auto_corr_sum += auto_correlation_coeff
        previous_arcoeff = auto_correlation_coeff

    return auto_corr_sum / variance


def compute_univariate_ess(chain, max_lag=None):
    r"""Estimate the effective sample size (ESS) of the given chain.

    The ESS is an estimate of the size of an iid sample with the same variance as the current sample.
    It is estimated using:

    .. math::

        ESS(X) = \frac{n}{\tau} = \frac{n}{1 + 2 * \sum_{k=1}^{m}{\rho_{k}}}

    where :math:`\rho_{k}` is estimated as:

    .. math::

        \hat{\rho}_{k} = \frac{E[(X_{t} - \mu)(X_{t + k} - \mu)]}{\sigma^{2}}

    Args:
        chain (ndarray): the chain for which to calculate the ESS, assumes a vector of length ``n`` samples
        max_lag (int): the maximum lag used in the variance calculations. If not given defaults to
            :math:`min(n/3, 1000)`.
    """
    variance = np.var(chain)

    if variance == 0:
        return 0

    return len(chain) / (1 + 2 * get_auto_correlation_time(chain, max_lag))


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


def estimate_multivariate_ess_sigma(samples, batch_size, nmr_offsets=None):
    r"""Calculates the Sigma matrix which is part of the multivariate ESS calculation.

    This implementation is based on the Matlab implementation found at: https://github.com/lacerbi/multiESS

    The Sigma matrix is defined as:

    .. math::

        \Sigma = \Lambda + 2 * \sum_{k=1}^{\infty}{Cov(Y_{1}, Y_{1+k})}

    Where :math:`Y` are our samples and :math:`\Lambda` is the covariance matrix of the samples.

    Args:
        samples (ndarray): the samples for which we compute the sigma matrix. Expects an pxn array with
            p the number of parameters and n the sample size
        batch_size (int): the batch size used in the approximation of the correlation covariance
        nmr_offsets (int): the number of offsets we will try in the approximation

    Returns:
        ndarray: an pxp array with p the number of parameters in the samples.

    References:
        Vats D, Flegal J, Jones G (2016). Multivariate Output Analysis for Markov Chain Monte Carlo.
        arXiv:1512.07713v2 [math.ST]
    """
    nmr_offsets = nmr_offsets or 10

    sample_means = np.mean(samples, axis=1)
    nmr_params, chain_length = samples.shape

    a = int(np.floor(chain_length / batch_size))
    sigma = np.zeros((nmr_params, nmr_params))

    offsets = np.unique(np.round(np.linspace(0, chain_length - a * batch_size, nmr_offsets)))

    for offset in offsets:
        Y = np.reshape(samples.T[np.array(offset + np.arange(0, a * batch_size), dtype=np.int), :],
                       [batch_size, a, nmr_params], order='F')
        Ybar = np.squeeze(np.mean(Y, axis=0))
        Z = Ybar - sample_means

        for i in range(0, a):
            sigma += Z[i, :, None].T * Z[i, :, None]

    return sigma * batch_size / (a - 1) / len(offsets)


def compute_multivariate_ess(samples, nmr_offsets=None, batch_size_generator=None, full_output=False):
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
        nmr_offsets (int): the number of offsets we use for the estimation of the :math:`\Sigma`
        batch_size_generator (MultiVariateESSBatchSizeGenerator): the batch size generator, tells us how many
            batches and of which size we use for estimating the minimum ESS.
        full_output (boolean): set to True to return the estimated :math:`\Sigma` and the optimal batch size.

    Returns:
        float or tuple: when full_output is set to True we return a tuple with the estimated multivariate ESS,
            the estimated :math:`\Sigma` matrix and the optimal batch size. When full_output is False (the default)
            we only return the ESS.

    References:
        Vats D, Flegal J, Jones G (2016). Multivariate Output Analysis for Markov Chain Monte Carlo.
        arXiv:1512.07713v2 [math.ST]
    """
    batch_size_generator = batch_size_generator or CubeRootSingleBatch()
    nmr_offsets = nmr_offsets or 10

    batch_sizes = batch_size_generator.get_batch_sizes(*samples.shape)

    nmr_params, chain_length = samples.shape
    nmr_batches = len(batch_sizes)

    det_lambda = det(np.cov(samples))

    ess_estimates = np.zeros(nmr_batches)
    sigma_estimates = np.zeros((nmr_params, nmr_params, nmr_batches))

    for i in range(0, nmr_batches):
        sigma = estimate_multivariate_ess_sigma(samples, int(batch_sizes[i]), nmr_offsets)
        ess = chain_length * (det_lambda / det(sigma)) ** (1.0 / nmr_params)

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


def multivariate_ess(samples, nmr_offsets=None, batch_size_generator=None):
    r"""Compute the multivariate Effective Sample Size for the samples of every problem.

    This essentially applies :func:`compute_multivariate_ess` to every problem in the samples.

    Args:
        samples (ndarray, dict or generator): either an matrix of shape (d, n, p) with d problems, n samples
            and p parameters or a dictionary with for every parameter a matrix with shape (d, n) or
            a generator function that yields sample arrays of shape (d, n).
        nmr_offsets (int): the number of offsets we use for the estimation of the :math:`\Sigma`
        batch_size_generator (MultiVariateESSBatchSizeGenerator): the batch size generator, tells us how many
            batches and of which size we use for estimating the minimum ESS.

    Returns:
        ndarray: the multivariate ESS per problem
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

    import multiprocessing
    p = multiprocessing.Pool()

    return np.array(list(p.imap(_MultivariateESSMultiProcessing(nmr_offsets, batch_size_generator),
                                samples_generator())))


class _MultivariateESSMultiProcessing(object):

    def __init__(self, nmr_offsets, batch_size_generator):
        """Used in the function :func:`multivariate_ess` to compute the multivariate ESS using multiprocessing."""
        self._nmr_offsets = nmr_offsets
        self._batch_size_generator = batch_size_generator

    def __call__(self, samples):
        return compute_multivariate_ess(samples, nmr_offsets=self._nmr_offsets,
                                        batch_size_generator=self._batch_size_generator)


class MultiVariateESSBatchSizeGenerator(object):
    """Objects of this class are used as input to the multivariate ESS function.

    The multivariate ESS function needs to have at least one batch size to use during the computations. More is also
    possible and then the lowest ESS of all batches is chosen. Objects of this class implement the logic behind
    choosing batch sizes.
    """

    def get_batch_sizes(self, nmr_params, chain_length):
        r"""Get the batch sizes to use for the calculation of the Effective Sample Size (ESS).

        This should return a list of batch sizes that the ESS calculation will use to determine :math:`\Sigma`

        Args:
            nmr_params (int): the number of parameters in the samples
            chain_length (int): the length of the chain

        Returns:
            list: the batches of the given sizes we will test in the ESS calculations
        """


class SquareRootSingleBatch(MultiVariateESSBatchSizeGenerator):
    r"""Returns :math:`\sqrt(n)`."""
    def get_batch_sizes(self, nmr_params, chain_length):
        return [np.floor(chain_length**(1/2.0))]


class CubeRootSingleBatch(MultiVariateESSBatchSizeGenerator):
    r"""Returns :math:`n^{1/3}`."""
    def get_batch_sizes(self, nmr_params, chain_length):
        return [np.floor(chain_length**(1/3.0))]


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

    def get_batch_sizes(self, nmr_params, chain_length):
        b_min = np.floor(chain_length**(1 / 4.0))
        b_max = np.max((np.floor(chain_length / np.max((nmr_params, 20))),
                        np.floor(chain_length**(1 / 2.0))))
        return list(np.unique(np.round(np.exp(np.linspace(np.log(b_min), np.log(b_max), self._nmr_batches)))))

