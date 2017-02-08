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


class UniformPrior(AbstractParameterPrior):
    """The uniform prior is always 1."""

    def get_prior_function(self):
        return '''
            #ifndef PRIOR_UNIFORM
            #define PRIOR_UNIFORM

            mot_float_type prior_uniform_prior(const mot_float_type value,
                                               const mot_float_type lower_bound,
                                               const mot_float_type upper_bound){
                return 1;
            }

            #endif //PRIOR_UNIFORM
        '''

    def get_prior_function_name(self):
        return 'prior_uniform_prior'


class ScalarReferencePrior(AbstractParameterPrior):
    """The uniform prior is always 1."""

    def get_prior_function(self):
        return '''
            #ifndef SCALAR_REFERENCE_PRIOR
            #define SCALAR_REFERENCE_PRIOR

            mot_float_type scalar_reference_prior(const mot_float_type value,
                                                  const mot_float_type lower_bound,
                                                  const mot_float_type upper_bound){
                if(value <= 0){
                    return 0;
                }
                return 1.0/value;
            }

            #endif //SCALAR_REFERENCE_PRIOR
        '''

    def get_prior_function_name(self):
        return 'scalar_reference_prior'


class UniformWithinBoundsPrior(AbstractParameterPrior):
    """This prior is 1 within the upper and lower bound of the parameter, 0 outside."""

    def get_prior_function(self):
        return '''
            #ifndef PRIOR_UNIFORM_WITHIN_BOUNDS
            #define PRIOR_UNIFORM_WITHIN_BOUNDS

            mot_float_type prior_uniform_within_bounds(const mot_float_type value,
                                                       const mot_float_type lower_bound,
                                                       const mot_float_type upper_bound){

                return (value < lower_bound || value > upper_bound) ? 0.0 : 1.0;
            }

            #endif //PRIOR_UNIFORM_WITHIN_BOUNDS
        '''

    def get_prior_function_name(self):
        return 'prior_uniform_within_bounds'


class AbsSinPrior(AbstractParameterPrior):
    """The fabs(sin(x)) prior."""

    def get_prior_function(self):
        return '''
            #ifndef PRIOR_ABSSIN
            #define PRIOR_ABSSIN

            mot_float_type prior_abs_sin(const mot_float_type value,
                                               const mot_float_type lower_bound,
                                               const mot_float_type upper_bound){
                return fabs(sin(value));
            }

            #endif //PRIOR_ABSSIN
        '''

    def get_prior_function_name(self):
        return 'prior_abs_sin'


class AbsSinHalfPrior(AbstractParameterPrior):
    """The fabs(sin(x)/2.0) prior."""

    def get_prior_function(self):
        return '''
            #ifndef PRIOR_ABSSIN_HALF
            #define PRIOR_ABSSIN_HALF

            mot_float_type prior_abs_sin_half(const mot_float_type value,
                                               const mot_float_type lower_bound,
                                               const mot_float_type upper_bound){
                return fabs(sin(value)/2.0);
            }

            #endif //PRIOR_ABSSIN_HALF
        '''

    def get_prior_function_name(self):
        return 'prior_abs_sin_half'


class NormalPDF(AbstractParameterPrior):

    def get_parameters(self):
        from mot.model_building.cl_functions.parameters import FreeParameter
        return [FreeParameter(CLDataType.from_string('mot_float_type'), 'mu', True, 0, -np.inf, np.inf,
                              sampling_prior=UniformPrior()),
                FreeParameter(CLDataType.from_string('mot_float_type'), 'sigma', False, 1, -np.inf, np.inf,
                              sampling_prior=UniformPrior())]

    def get_prior_function(self):
        return '''
            #ifndef PRIOR_NORMALPDF
            #define PRIOR_NORMALPDF

            mot_float_type prior_normal_pdf(const mot_float_type value,
                                            const mot_float_type lower_bound,
                                            const mot_float_type upper_bound,
                                            const mot_float_type mu,
                                            const mot_float_type sigma){

                return exp(-pown(value - mu, 2) / (2 * pown(sigma, 2)))
                        / (sigma * sqrt(2 * M_PI));

            }

            #endif //PRIOR_NORMALPDF
        '''

    def get_prior_function_name(self):
        return 'prior_normal_pdf'


class BetaPDF(AbstractParameterPrior):

    def get_parameters(self):
        from mot.model_building.cl_functions.parameters import FreeParameter
        return [FreeParameter(CLDataType.from_string('mot_float_type'), 'alpha', True, 1, 0, np.inf,
                              sampling_prior=UniformWithinBoundsPrior()),
                FreeParameter(CLDataType.from_string('mot_float_type'), 'beta', False, 0.5, 0, 10,
                              sampling_prior=ScalarReferencePrior(),
                              sampling_proposal=GaussianProposal(0.01))]

    def get_prior_function(self):
        return '''
            #ifndef PRIOR_BETAPDF
            #define PRIOR_BETAPDF

            mot_float_type prior_beta_pdf(const mot_float_type value,
                                          const mot_float_type lower_bound,
                                          const mot_float_type upper_bound,
                                          const mot_float_type alpha,
                                          const mot_float_type beta){

                if(value <= 0 || value >= 1){
                    return 0;
                }

                return (tgamma(alpha + beta) * pow(1 - value, beta - 1) * pow(value, alpha - 1))
                            / (tgamma(alpha) * tgamma(beta));
            }

            #endif //PRIOR_BETAPDF
        '''

    def get_prior_function_name(self):
        return 'prior_beta_pdf'
