import numpy as np
from mot.cl_data_type import SimpleCLDataType
from mot.model_building.parameter_functions.proposals import GaussianProposal

__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ParameterPrior(object):
    """The priors are used during model sampling.

    These priors are not in log space, we take the log in the model builder.

    They indicate the a priori information one has about a parameter.
    """

    def get_prior_function(self):
        """Get the prior function as a CL string. This should include include guards (#ifdef's).

        This should follow the signature:

        .. code-block: c

            mot_float_type <prior_fname>(mot_float_type parent_parameter,
                                         mot_float_type lower_bound,
                                         mot_float_type upper_bound,
                                         <sub-parameters>)

        That is, the parent parameter and it lower and upper bound is given next to the optional parameters
        defined in this prior.

        Returns:
            str: The cl function
        """
        raise NotImplementedError()

    def get_prior_function_name(self):
        """Get the name of the prior function call.

         This is used by the model builder to construct the call to the prior function.

         Returns:
            str: name of the function
        """
        raise NotImplementedError()

    def get_parameters(self):
        """Get the additional parameters featured in this prior.

        This can return a list of additional parameters to be used in the model function.

        Returns:
            list of CLFunctionParameter: the list of function parameters to be added to the list of
                parameters of the enclosing model.
        """
        raise NotImplementedError()


class SimplePrior(ParameterPrior):

    def __init__(self, prior_body, prior_name, prior_params=None, cl_preamble=None):
        """A prior template function.

        Args:
            prior_body (str): the body of the prior
            prior_name (str): the name of this prior function
            prior_params (list): additional parameters for this prior
            preamble (str): optional C code loaded before the function definition.
        """
        self._prior_body = prior_body
        self._prior_name = prior_name
        self._prior_params = prior_params or []
        self._cl_preamble = cl_preamble

    def get_parameters(self):
        return self._prior_params

    def get_prior_function(self):
        params = ['value', 'lower_bound', 'upper_bound']
        params.extend(p.name for p in self._prior_params)
        params = ['const mot_float_type {}'.format(v) for v in params]

        return '''
            {cl_preamble}

            #ifndef {include_guard_name}
            #define {include_guard_name}

            mot_float_type {function_name}({params}){{
                {prior_body}
            }}

            #endif //{include_guard_name}
        '''.format(include_guard_name='PRIOR_{}'.format(self._prior_name.upper()), function_name=self._prior_name,
                   prior_body=self._prior_body, params=', '.join(params), cl_preamble=self._cl_preamble or '')

    def get_prior_function_name(self):
        return self._prior_name


class AlwaysOne(SimplePrior):

    def __init__(self):
        """The uniform prior is always 1. :math:`P(v) = 1` """
        super(AlwaysOne, self).__init__('return 1;', 'uniform')


class ReciprocalPrior(SimplePrior):

    def __init__(self):
        """The reciprocal of the current value. :math:`P(v) = 1/v` """
        body = '''
            if(value <= 0){
                return 0;
            }
            return 1.0/value;
        '''
        super(ReciprocalPrior, self).__init__(body, 'reciprocal')


class UniformWithinBoundsPrior(SimplePrior):

    def __init__(self):
        """This prior is 1 within the upper and lower bound of the parameter, 0 outside."""
        super(UniformWithinBoundsPrior, self).__init__(
            'return value >= lower_bound && value <= upper_bound;',
            'uniform_within_bounds')


class AbsSinPrior(SimplePrior):

    def __init__(self):
        """Angular prior: :math:`P(v) = |\\sin(v)|`"""
        super(AbsSinPrior, self).__init__('return fabs(sin(value));', 'abs_sin')


class AbsSinHalfPrior(SimplePrior):

    def __init__(self):
        """Angular prior: :math:`P(v) = |\\sin(x)/2.0|`"""
        super(AbsSinHalfPrior, self).__init__('return fabs(sin(value)/2.0);', 'abs_sin_half')


class VagueGammaPrior(SimplePrior):

    def __init__(self):
        """The vague gamma prior is meant as a proper uniform prior.

        Lee & Wagenmakers:

            The practice of assigning Gamma(0.001, 0.001) priors on precision parameters is theoretically motivated by
            scale invariance arguments, meaning that priors are chosen so that changing the measurement
            scale of the data does not affect inference.
            The invariant prior on precision lambda corresponds to a uniform distribution on log sigma,
            that is, rho(sigma^2) prop. to. 1/sigma^2, or a Gamma(a -> 0, b -> 0) distribution.
            This invariant prior distribution, however, is improper (i.e., the area under the curve is unbounded),
            which means it is not really a distribution, but the limit of a sequence of distributions
            (see Jaynes, 2003). WinBUGS requires the use of proper distributions,
            and the Gamma(0.001, 0.001) prior is intended as a proper approximation to the theoretically
            motivated improper prior. This raises the issue of whether inference is sensitive to the essentially
            arbitrary value 0.001, and it is sometimes the case that using other small values such as 0.01 or 0.1
            leads to more stable sampling
            in WinBUGS.

            -- Lee & Wagenmakers, Bayesian Cognitive Modeling, 2014, Chapter 4, Box 4.1

        While this is not WinBUGS and improper priors are allowed in MOT, it is still useful to have this prior
        in case people desire proper priors.
        """
        body = '''
            float kappa = 0.001;
            float theta = 1/0.001;

            return (1.0 / (tgamma(kappa) * pow(theta, kappa))) * pow(value, kappa - 1) * exp(- value / theta);
        '''
        super(VagueGammaPrior, self).__init__(body, 'vague_gamma_prior', [])


class NormalPDF(SimplePrior):

    def __init__(self):
        r"""Normal PDF on the given value: :math:`P(v) = N(v; \mu, \sigma)`"""
        from mot.model_building.parameters import FreeParameter
        params = [FreeParameter(SimpleCLDataType.from_string('mot_float_type'), 'mu', True, 0, -np.inf, np.inf,
                                sampling_prior=AlwaysOne()),
                  FreeParameter(SimpleCLDataType.from_string('mot_float_type'), 'sigma', True, 1, -np.inf, np.inf,
                                sampling_prior=AlwaysOne())]

        super(NormalPDF, self).__init__(
            'return exp(-pown(value - mu, 2) / (2 * pown(sigma, 2))) / (sigma * sqrt(2 * M_PI));',
            'normal_pdf',
            params)


class AxialNormalPDF(SimplePrior):

    def __init__(self):
        r"""The axial normal PDF is a Normal distribution wrapped around 0 and :math:`\pi`.

        It's PDF is given by:

        .. math::

            f(\theta; a, b) = \frac{\cosh(a\sin \theta + b\cos \theta)}{\pi I_{0}(\sqrt{a^{2} + b^{2}})}

        where in this implementation :math:`a` and :math:`b` are parameterized with the input variables
        :math:`\mu` and :math:`\sigma` using:

        .. math::

            \begin{align*}
            \kappa &= \frac{1}{\sigma^{2}} \\
            a &= \kappa * \sin \mu \\
            b &= \kappa * \cos \mu
            \end{align*}

        References:
            Barry C. Arnold, Ashis SenGupta (2006). Probability distributions and statistical inference for axial data.
            Environmental and Ecological Statistics, volume 13, issue 3, pages 271-285.
        """
        from mot.model_building.parameters import FreeParameter
        from mot.library_functions import Bessel, Trigonometrics

        params = [FreeParameter(SimpleCLDataType.from_string('mot_float_type'), 'mu', True, 0, -np.inf, np.inf,
                                sampling_prior=AlwaysOne()),
                  FreeParameter(SimpleCLDataType.from_string('mot_float_type'), 'sigma', True, 1, -np.inf, np.inf,
                                sampling_prior=AlwaysOne())]

        super(AxialNormalPDF, self).__init__(
            '''
                float kappa = 1.0 / pown(sigma, 2);
                float a = kappa * sin(mu);
                float b = kappa * cos(mu);

                return exp(log_cosh(a * sin(value) + b * cos(value))
                            - log_bessel_i0(sqrt(pown(a, 2) + pown(b, 2)))
                            - log(M_PI) );
            ''',
            'axial_normal_pdf',
            params,
            cl_preamble=Bessel().get_cl_code() + '\n' + Trigonometrics().get_cl_code())


class ARDBeta(SimplePrior):

    def __init__(self):
        r"""This is a collapsed form of the Beta PDF meant for use in Automatic Relevance Detection sampling.

        In this prior the ``alpha`` parameter of the Beta prior is set to 1 which simplifies the equation.
        The parameter ``beta`` is still free and can be changed as desired.

        The implemented prior is:

        .. math::

            B(x; 1, \beta) = \beta * (1 - x)^{\beta - 1}

        """
        from mot.model_building.parameters import FreeParameter
        params = [FreeParameter(SimpleCLDataType.from_string('mot_float_type'), 'beta', False, 1, 1e-4, 1000,
                                sampling_prior=ReciprocalPrior(),
                                sampling_proposal=GaussianProposal(0.01))]

        body = '''
            if(value < 0 || value > 1){
                return 0;
            }
            return beta * pow(1 - value, beta - 1);
        '''
        super(ARDBeta, self).__init__(body, 'ard_beta_pdf', params)


class ARDGaussian(SimplePrior):

    def __init__(self):
        """This is a Gaussian prior meant for use in Automatic Relevance Detection sampling.

        This uses a Gaussian prior with mean at zero and a standard deviation determined by the ``alpha`` parameter
        with the relationship :math:`\sigma = 1/\\sqrt(\\alpha)`.
        """
        from mot.model_building.parameters import FreeParameter
        params = [FreeParameter(SimpleCLDataType.from_string('mot_float_type'), 'alpha', False, 8, 1e-5, 1e4,
                                sampling_prior=UniformWithinBoundsPrior(),
                                sampling_proposal=GaussianProposal(20))]

        body = '''
            if(value < 0 || value > 1){
                return 0;
            }
            mot_float_type sigma = 1.0/sqrt(alpha);
            return exp(-pown(value, 2) / (2 * pown(sigma, 2))) / (sigma * sqrt(2 * M_PI));
        '''
        super(ARDGaussian, self).__init__(body, 'ard_beta_pdf', params)
