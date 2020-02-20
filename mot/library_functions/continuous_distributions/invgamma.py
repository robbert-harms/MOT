from mot.library_functions.base import SimpleCLLibrary

__author__ = 'Robbert Harms'
__date__ = '2018-05-07'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'

from mot.library_functions.continuous_distributions.gamma import igamc, igamci


class invgamma_pdf(SimpleCLLibrary):
    def __init__(self):
        r"""Computes the inverse-Gamma probability density function using the shape and scale parameterization.

        This computes the inverse-Gamma PDF as:

        .. math::

            f(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} (1/x)^{\alpha + 1}\exp\left(-\beta/x\right)

        With :math:`x` the desired position, :math:`\alpha` the shape and :math:`\beta` the scale.
        """
        super().__init__('''
            double invgamma_pdf(double x, double shape, double scale){
                return exp(invgamma_logpdf(x, shape, scale));
            }
        ''', dependencies=[invgamma_logpdf()])


class invgamma_logpdf(SimpleCLLibrary):
    def __init__(self):
        r"""Computes the log of the inverse-Gamma probability density function with shape and scale parameterization.
        """
        super().__init__('''
            double invgamma_logpdf(double x, double shape, double scale){
                double _x = x / scale;
                return -(shape+1) * log(_x) - lgamma(shape) - 1.0/_x - log(scale);
            }
        ''')


class invgamma_cdf(SimpleCLLibrary):
    def __init__(self):
        r"""Calculate the Cumulative Distribution Function of the inverse-Gamma function.

        This implementation is copied from SciPy (09-01-2020).

        Function arguments:

         * shape: the shape parameter of the gamma distribution (often denoted :math:`\alpha`)
         * scale: the scale parameter of the gamma distribution (often denoted :math:`\beta`)
        """
        super().__init__('''
            double invgamma_cdf(double x, double shape, double scale){
                return igamc(shape, scale / x);
            }
        ''', dependencies=(igamc(),))


class invgamma_ppf(SimpleCLLibrary):

    def __init__(self):
        """Computes the inverse of the cumulative distribution function of the inverse-Gamma distribution.

        This is the inverse of the inverse-Gamma CDF.
        """
        super().__init__('''
            double invgamma_ppf(double y, double shape, double scale){
                return scale / igamci(shape, y);
            }
        ''', dependencies=(igamci(),))
