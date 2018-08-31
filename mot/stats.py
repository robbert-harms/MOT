import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import norm
import scipy.integrate

from mot.lib.cl_function import SimpleCLFunction
from mot.lib.kernel_data import Array, Zeros, Scalar
from mot.lib.utils import is_scalar, multiprocess_mapping

__author__ = 'Robbert Harms'
__date__ = '2017-11-01'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


def fit_gaussian(samples, ddof=0):
    """Calculates the mean and the standard deviation of the given samples.

    Args:
        samples (ndarray): a one or two dimensional array. If one dimensional we calculate the fit using all
            values. If two dimensional, we fit the Gaussian for every set of samples over the first dimension.
        ddof (int): the difference degrees of freedom in the std calculation. See numpy.
    """
    if len(samples.shape) == 1:
        return np.mean(samples), np.std(samples, ddof=ddof)
    return np.mean(samples, axis=1), np.std(samples, axis=1, ddof=ddof)


def fit_circular_gaussian(samples, high=np.pi, low=0):
    """Compute the circular mean for samples in a range

    Args:
        samples (ndarray): a one or two dimensional array. If one dimensional we calculate the fit using all
            values. If two dimensional, we fit the Gaussian for every set of samples over the first dimension.
        high (float): The maximum wrap point
        low (float): The minimum wrap point
    """
    cl_func = SimpleCLFunction.from_string('''
        void compute(global mot_float_type* samples,
                     global mot_float_type* means,
                     global mot_float_type* stds,
                     int nmr_samples,
                     int low,
                     int high){ 
        
            double cos_mean = 0;
            double sin_mean = 0;
            double ang;

            for(uint i = 0; i < nmr_samples; i++){
                ang = (samples[i] - low)*2*M_PI / (high - low);

                cos_mean += (cos(ang) - cos_mean) / (i + 1);
                sin_mean += (sin(ang) - sin_mean) / (i + 1);
            }

            double R = hypot(cos_mean, sin_mean);
            if(R > 1){
                R = 1;
            }

            double stds = 1/2. * sqrt(-2 * log(R));

            double res = atan2(sin_mean, cos_mean);
            if(res < 0){
                 res += 2 * M_PI;
            }

            *(means) = res*(high - low)/2.0/M_PI + low;
            *(stds) = ((high - low)/2.0/M_PI) * sqrt(-2*log(R));
        }
    ''')

    def run_cl(samples):
        data = {'samples': Array(samples, 'mot_float_type'),
                'means': Zeros(samples.shape[0], 'mot_float_type'),
                'stds': Zeros(samples.shape[0], 'mot_float_type'),
                'nmr_samples': Scalar(samples.shape[1]),
                'low': Scalar(low),
                'high': Scalar(high),
                }

        cl_func.evaluate(data, samples.shape[0])
        return data['means'].get_data(), data['stds'].get_data()

    if len(samples.shape) == 1:
        mean, std = run_cl(samples[None, :])
        return mean[0], std[0]
    return run_cl(samples)


def fit_truncated_gaussian(samples, lower_bounds, upper_bounds):
    """Fits a truncated gaussian distribution on the given samples.

    This will do a maximum likelihood estimation of a truncated Gaussian on the provided samples, with the
    truncation points given by the lower and upper bounds.

    Args:
        samples (ndarray): a one or two dimensional array. If one dimensional we fit the truncated Gaussian on all
            values. If two dimensional, we calculate the truncated Gaussian for every set of samples over the
            first dimension.
        lower_bounds (ndarray or float): the lower bound, either a scalar or a lower bound per problem (first index of
            samples)
        upper_bounds (ndarray or float): the upper bound, either a scalar or an upper bound per problem (first index of
            samples)

    Returns:
        mean, std: the mean and std of the fitted truncated Gaussian
    """
    if len(samples.shape) == 1:
        return _TruncatedNormalFitter()((samples, lower_bounds, upper_bounds))

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

            yield (samples[ind], lower_bound, upper_bound)

    results = np.array(multiprocess_mapping(_TruncatedNormalFitter(), item_generator()))
    return results[:, 0], results[:, 1]


def gaussian_overlapping_coefficient(means_0, stds_0, means_1, stds_1, lower=None, upper=None):
    """Compute the overlapping coefficient of two Gaussian continuous_distributions.

    This computes the :math:`\int_{-\infty}^{\infty}{\min(f(x), g(x))\partial x}` where
    :math:`f \sim \mathcal{N}(\mu_0, \sigma_0^{2})` and :math:`f \sim \mathcal{N}(\mu_1, \sigma_1^{2})` are normally
    distributed variables.

    This will compute the overlap for each element in the first dimension.

    Args:
        means_0 (ndarray): the set of means of the first distribution
        stds_0 (ndarray): the set of stds of the fist distribution
        means_1 (ndarray): the set of means of the second distribution
        stds_1 (ndarray): the set of stds of the second distribution
        lower (float): the lower limit of the integration. If not set we set it to -inf.
        upper (float): the upper limit of the integration. If not set we set it to +inf.
    """
    if lower is None:
        lower = -np.inf
    if upper is None:
        upper = np.inf

    def point_iterator():
        for ind in range(means_0.shape[0]):
            yield np.squeeze(means_0[ind]), np.squeeze(stds_0[ind]), np.squeeze(means_1[ind]), np.squeeze(stds_1[ind])

    return np.array(list(multiprocess_mapping(_ComputeGaussianOverlap(lower, upper), point_iterator())))


def deviance_information_criterions(mean_posterior_lls, ll_per_sample):
    r"""Calculates the Deviance Information Criteria (DIC) using three methods.

    This returns a dictionary returning the ``DIC_2002``, the ``DIC_2004`` and the ``DIC_Ando_2011`` method.
    The first is based on Spiegelhalter et al (2002), the second based on Gelman et al. (2004) and the last on
    Ando (2011). All cases differ in how they calculate model complexity, i.e. the effective number of parameters
    in the model. In all cases the model with the smallest DIC is preferred.

    All these DIC methods measure fitness using the deviance, which is, for a likelihood :math:`p(y | \theta)`
    defined as:

    .. math::

        D(\theta) = -2\log p(y|\theta)

    From this, the posterior mean deviance,

    .. math::

        \bar{D} = \mathbb{E}_{\theta}[D(\theta)]

    is then used as a measure of how well the model fits the data.

    The complexity, or measure of effective number of parameters, can be measured in see ways, see
    Spiegelhalter et al. (2002), Gelman et al (2004) and Ando (2011). The first method calculated the parameter
    deviance as:

    .. math::
        :nowrap:

        \begin{align}
        p_{D} &= \mathbb{E}_{\theta}[D(\theta)] - D(\mathbb{E}[\theta)]) \\
              &= \bar{D} - D(\bar{\theta})
        \end{align}

    i.e. posterior mean deviance minus the deviance evaluated at the posterior mean of the parameters.

    The second method calculated :math:`p_{D}` as:

    .. math::

        p_{D} = p_{V} = \frac{1}{2}\hat{var}(D(\theta))

    i.e. half the variance of the deviance is used as an estimate of the number of free parameters in the model.

    The third method calculates the parameter deviance as:

    .. math::

        p_{D} = 2 \cdot (\bar{D} - D(\bar{\theta}))

    That is, twice the complexity of that of the first method.

    Finally, the DIC is (for all cases) defined as:

    .. math::

        DIC = \bar{D} + p_{D}

    Args:
        mean_posterior_lls (ndarray): a 1d matrix containing the log likelihood for the average posterior
            point estimate. That is, the single log likelihood of the average parameters.
        ll_per_sample (ndarray): a (d, n) array with for d problems the n log likelihoods.
            This is the log likelihood per sample.

    Returns:
        dict: a dictionary containing the ``DIC_2002``, the ``DIC_2004`` and the ``DIC_Ando_2011`` information
            criterion maps.
    """
    mean_deviance = -2 * np.mean(ll_per_sample, axis=1)
    deviance_at_mean = -2 * mean_posterior_lls

    pd_2002 = mean_deviance - deviance_at_mean
    pd_2004 = np.var(ll_per_sample, axis=1) / 2.0

    return {'DIC_2002': np.nan_to_num(mean_deviance + pd_2002),
            'DIC_2004': np.nan_to_num(mean_deviance + pd_2004),
            'DIC_Ando_2011': np.nan_to_num(mean_deviance + 2 * pd_2002)}


class _ComputeGaussianOverlap:

    def __init__(self, lower, upper):
        """Helper routine for :func:`gaussian_overlapping_coefficient`.

        This calculates the overlap between two Normal distribution by taking the integral from lower to upper
        over min(f, g).
        """
        self.lower = lower
        self.upper = upper

    def __call__(self, data):
        """Compute the overlap.

        This expects data to be a tuple consisting of :math:`\mu_0, \sigma_0, \mu_1, \sigma_1`.

        Returns:
            float: the overlap between the two Gaussians.
        """
        m0, std0, m1, std1 = data

        def overlap_func(x):
            fx = norm.pdf(x, m0, std0)
            gx = norm.pdf(x, m1, std1)
            return min(fx, gx)
        return scipy.integrate.quad(overlap_func, self.lower, self.upper)[0]


class _TruncatedNormalFitter:

    def __call__(self, item):
        """Fit the mean and std of the truncated normal to the given samples.

        Helper function of :func:`fit_truncated_gaussian`.
        """
        samples, lower_bound, upper_bound = item
        scaling_factor = 10 ** -np.round(np.log10(np.mean(samples)))
        result = minimize(_TruncatedNormalFitter.truncated_normal_log_likelihood,
                          np.array([np.mean(samples), np.std(samples)]) * scaling_factor,
                          args=(lower_bound * scaling_factor,
                                upper_bound * scaling_factor,
                                samples * scaling_factor),
                          method='L-BFGS-B',
                          jac=_TruncatedNormalFitter.truncated_normal_ll_gradient,
                          bounds=[(lower_bound * scaling_factor, upper_bound * scaling_factor), (0, None)]
        )
        return result.x / scaling_factor

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
        if sigma == 0:
            return np.inf
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
        if params[1] == 0:
            return np.array([np.inf, np.inf])

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
