__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractParameterProposal(object):
    """The parameter proposals are meant for use during sampling.

    They indicate how a proposal should be generated for the given parameter.

    The proposals can allow for dynamic updates to better adapt to the landscape during sampling.
    """

    @property
    def is_symmetric(self):
        """Check if this proposal is symmetric. That is, if q(x|y) == q(y|x)."""
        return True

    def get_parameters(self):
        """The proposal parameters.

        Returns:
            list of ProposalParameter: the proposal parameter objects used by this proposal
        """

    def get_proposal_function(self):
        """Get the proposal function as a CL string. This should include include guards (#ifdef's).

        This should follow the signature:
        double <proposal_fname>(double current, ranluxcl_state_t* ranlux, <additional_parameters>)

        That is, it can have more than two parameter, but the first two are obligatory. The additional parameters
        are defined by the get_parameters function of this python class.

        Returns:
            str: The cl function
        """

    def get_proposal_function_name(self):
        """Get the name of the proposal function call.

         This is used by the model builder to construct the call to the proposal function.

         Returns:
            str: name of the function
        """

    def get_proposal_logpdf_function(self):
        """Get the proposal pdf function as a CL string. This should include include guards.

        This should follow the signature:
            double <proposal_pdf_fname>(double proposal, double current, <additional_parameters>)

        Returns:
            str: The proposal log pdf function as a CL string
        """

    def get_proposal_logpdf_function_name(self):
        """Get the name of the proposal logpdf function call.

         This is used by the model builder to construct the call to the proposal logpdf function.

         Returns:
            str: name of the function
        """


class ProposalParameter(object):

    def __init__(self, default_value, adaptable):
        """Container class for parameters of a proposal function.

        Args:
            default_value (double): the parameter value
            adaptable (boolean): if this parameter is adaptable during sampling

        Attributes:
            default_value (double): the parameter value
            adaptable (boolean): if this parameter is adaptable
        """
        self.default_value = default_value
        self.adaptable = adaptable

    def get_parameter_update_function(self):
        """Get the parameter update function used to update this proposal parameter.

        Returns:
            A function with the signature:
                mot_float_type <func_name>(const mot_float_type current_value,
                                           const uint acceptance_counter,
                                           const uint jump_counter)

            Where current value is the current value for this proposal parameter, acceptance counter is the
            number of accepted steps and jump counter the number of jumps.
        """
        return '''
            #ifndef PROPOSAL_DEFAULT_PARAMETER_UPDATE
            #define PROPOSAL_DEFAULT_PARAMETER_UPDATE

            mot_float_type proposal_default_parameter_update(const mot_float_type current_value,
                                                             const uint acceptance_counter,
                                                             const uint jump_counter){
                return min(current_value *
                            sqrt( (mot_float_type)(acceptance_counter+1) /
                                  ((jump_counter - acceptance_counter) + 1)
                            ),
                           (mot_float_type)1e10);
            }

            #endif //PROPOSAL_DEFAULT_PARAMETER_UPDATE
        '''

    def get_parameter_update_function_name(self):
        return 'proposal_default_parameter_update'


class GaussianProposal(AbstractParameterProposal):

    def __init__(self, std=1.0, adaptable=True):
        """Create a new proposal function using a Gaussian distribution with the given scale.

        Args:
            gaussian_scale (float): The scale of the Gaussian distribution.
            adaptable (boolean): If this proposal is adaptable during sampling

        Attributes:
            gaussian_scale (float): The scale of the Gaussian distribution.
        """
        self._parameters = [ProposalParameter(std, adaptable)]

    def get_parameters(self):
        return self._parameters

    def get_proposal_function(self):
        return '''
            #ifndef PROP_GAUSSIANPROPOSAL_CL
            #define PROP_GAUSSIANPROPOSAL_CL

            mot_float_type proposal_gaussianProposal(mot_float_type current,
                                                     ranluxcl_state_t* const ranluxclstate,
                                                     mot_float_type std){
                return fma(std, (mot_float_type)ranluxcl_gaussian(ranluxclstate), current);
            }

            #endif //PROP_GAUSSIANPROPOSAL_CL
        '''

    def get_proposal_function_name(self):
        return 'proposal_gaussianProposal'

    def get_proposal_logpdf_function(self):
        return '''
            #ifndef PROP_GAUSSIANPROPOSALLOGPDF_CL
            #define PROP_GAUSSIANPROPOSALLOGPDF_CL

            mot_float_type proposal_gaussianProposalLogPDF(mot_float_type x,
                                                           mot_float_type mu,
                                                           mot_float_type std){
                return log(std * sqrt(2 * M_PI)) - (((x - mu) * (x - mu)) / (2 * std * std));
            }

            #endif //PROP_GAUSSIANPROPOSALLOGPDF_CL
        '''

    def get_proposal_logpdf_function_name(self):
        return 'proposal_gaussianProposalLogPDF'

