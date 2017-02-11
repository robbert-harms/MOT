import numpy as np
from mot.cl_data_type import CLDataType
from mot.model_building.parameter_functions.proposals import GaussianProposal

__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractParameterPrior(object):
    """The priors are used during model sampling.

    These priors should be in the

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

    def get_prior_function_name(self):
        """Get the name of the prior function call.

         This is used by the model builder to construct the call to the prior function.

         Returns:
            str: name of the function
        """

    def get_parameters(self):
        """Get the additional parameters featured in this prior.

        This can return a list of additional parameters to be used in the model function.

        Returns:
            list of CLFunctionParameter: the list of function parameters to be added to the list of
                parameters of the enclosing model.
        """
        return []


class SimplePrior(AbstractParameterPrior):

    def __init__(self, prior_body, prior_name, prior_params=None):
        self._prior_body = prior_body
        self._prior_name = prior_name
        self._prior_params = prior_params or []

    def get_parameters(self):
        return self._prior_params

    def get_prior_function(self):
        params = ['value', 'lower_bound', 'upper_bound']
        params.extend(p.name for p in self._prior_params)
        params = ['const mot_float_type {}'.format(v) for v in params]

        return '''
            #ifndef {include_guard_name}
            #define {include_guard_name}

            mot_float_type {function_name}({params}){{
                {prior_body}
            }}

            #endif //{include_guard_name}
        '''.format(include_guard_name='PRIOR_{}'.format(self._prior_name.upper()), function_name=self._prior_name,
                   prior_body=self._prior_body, params=', '.join(params))

    def get_prior_function_name(self):
        return self._prior_name


class AlwaysOne(SimplePrior):

    def __init__(self):
        """The uniform prior is always 1. ``P(v) = 1`` """
        super(AlwaysOne, self).__init__('return 1;', 'uniform')


class ReciprocalPrior(SimplePrior):

    def __init__(self):
        """The reciprocal of the current value. ``P(v) = 1/v`` """
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
            'return (value < lower_bound || value > upper_bound) ? 0.0 : 1.0;',
            'uniform_within_bounds')


class AbsSinPrior(SimplePrior):

    def __init__(self):
        """Angular prior: ``P(v) = |sin(v)|``"""
        super(AbsSinPrior, self).__init__('return fabs(sin(value));', 'abs_sin')


class AbsSinHalfPrior(SimplePrior):

    def __init__(self):
        """Angular prior: ``P(v) = |sin(x)/2.0|``"""
        super(AbsSinHalfPrior, self).__init__('return fabs(sin(value)/2.0);', 'abs_sin_half')


class NormalPDF(SimplePrior):

    def __init__(self):
        """Normal PDF on the given value: ``P(v) = N(v; mu, sigma)``"""
        from mot.model_building.cl_functions.parameters import FreeParameter
        params = [FreeParameter(CLDataType.from_string('mot_float_type'), 'mu', True, 0, -np.inf, np.inf,
                                sampling_prior=AlwaysOne()),
                  FreeParameter(CLDataType.from_string('mot_float_type'), 'sigma', False, 1, -np.inf, np.inf,
                                sampling_prior=AlwaysOne())]

        super(NormalPDF, self).__init__(
            'return exp(-pown(value - mu, 2) / (2 * pown(sigma, 2))) / (sigma * sqrt(2 * M_PI));',
            'normal_pdf',
            params)


class ARDBetaPDF(SimplePrior):

    def __init__(self):
        """This is a collapsed form of the Beta PDF meant for use in Automatic Relevance Detection sampling.

        In this prior the ``alpha`` parameter of the Beta prior is locked to 1 which simplifies the equation.
        The beta parameter is still free and can be changed as desired.

        The implemented prior is ``beta * pow(1 - value, beta - 1)``.

        """
        from mot.model_building.cl_functions.parameters import FreeParameter
        params = [FreeParameter(CLDataType.from_string('mot_float_type'), 'beta', False, 1, 1, 1000,
                                sampling_prior=ReciprocalPrior(),
                                sampling_proposal=GaussianProposal(0.01))]

        body = '''
            if(value <= 0 || value >= 1){
                return 0;
            }
            return beta * pow(1 - value, beta - 1);
        '''

        super(ARDBetaPDF, self).__init__(body, 'ard_beta_pdf', params)
