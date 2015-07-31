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
            #ifndef DMRIPROP_GAUSSIANPROPOSAL_CL
            #define DMRIPROP_GAUSSIANPROPOSAL_CL

            double dmriproposal_gaussianProposal(const double current, ranluxcl_state_t* const ranluxclstate,
                                                 const double std){
                return current + std * ranluxcl_gaussian(ranluxclstate);
            }

            #endif //DMRIPROP_GAUSSIANPROPOSAL_CL
        '''

    def get_proposal_function_name(self):
        return 'dmriproposal_gaussianProposal'

    def get_proposal_logpdf_function(self):
        return '''
            #ifndef DMRIPROP_GAUSSIANPROPOSALLOGPDF_CL
            #define DMRIPROP_GAUSSIANPROPOSALLOGPDF_CL

            double dmriproposal_gaussianProposalLogPDF(const double x, const double mu, const double std){
                return log(M_2_SQRTPI / std) + (-0.5 * pown((x - mu) / std, 2));
            }

            #endif //DMRIPROP_GAUSSIANPROPOSALLOGPDF_CL
        '''

    def get_proposal_logpdf_function_name(self):
        return 'dmriproposal_gaussianProposalLogPDF'

