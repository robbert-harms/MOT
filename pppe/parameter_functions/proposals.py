__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractParameterProposal(object):
    """The parameter proposals are meant for use during sampling.

    They indicate how a proposal should be generated for the given parameter.
    """

    @property
    def is_symmetric(self):
        """Check if this proposal is symmetric. That is, if q(x|y) == q(y|x)."""
        return True

    def get_proposal_function(self):
        """Get the proposal function as a CL string. This should include include guards (ifdef's).

        Returns:
            str: The cl function
        """

    def get_proposal_call(self, parameter, param_name, ranlux_name):
        """Get the function call.

        This should follow the signature:
        double <proposal_fname>(double <param_name>, ranluxcl_state_t* <ranlux_name>, ...)
        That is, it can have more than two parameter, but the first two should be replaceable by the composite model.

        Args:
            parameter (CLFunctionParameter): The context parameter
            param_name (str): The parameter name for in CL
            ranlux_name (str): The ranlux name for in CL

        Returns:
            str: The proposal calling function.
        """

    def get_proposal_logpdf_function(self):
        """Get the proposal pdf function as a CL string. This should include include guards.

        Returns:
            str: The proposal log pdf function as a CL string
        """

    def get_proposal_logpdf_call(self, proposal_name, current_name):
        """Get the function call to the proposal log pdf function.

        This should follow the signature:
            double <proposal_pdf_fname>(<proposal_name>, <current_name>, ...)

        Args:
            proposal_name (str): The proposal name for in CL
            current_name (str): The current variable name for in CL

        Returns:
            str: The proposal log pdf calling function.
        """


class GaussianProposal(AbstractParameterProposal):

    def __init__(self, gaussian_scale=1.0):
        """Create a new proposal function using a Gaussian distribution with the given scale.

        Args:
            gaussian_scale (float): The scale of the Gaussian distribution.

        Attributes:
            gaussian_scale (float): The scale of the Gaussian distribution.
        """
        self.gaussian_scale = float(gaussian_scale)

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

    def get_proposal_call(self, parameter, param_name, ranlux_name):
        # The standard deviation is in the proposal call to enable multiple proposal calls with different stds.
        return 'dmriproposal_gaussianProposal(' + param_name + ', ' + ranlux_name + ', ' + \
               repr(self.gaussian_scale) + ')'

    def get_proposal_logpdf_function(self):
        return '''
            #ifndef DMRIPROP_GAUSSIANPROPOSALLOGPDF_CL
            #define DMRIPROP_GAUSSIANPROPOSALLOGPDF_CL

            double dmriproposal_gaussianProposalLogPDF(const double x, const double mu, const double sigma){
                return log(M_2_SQRTPI / sigma) + (-0.5 * pown((x - mu) / sigma, 2));
            }

            #endif //DMRIPROP_GAUSSIANPROPOSALLOGPDF_CL
        '''

    def get_proposal_logpdf_call(self, proposal_name, current_name):
        return 'dmriproposal_gaussianProposalLogPDF(' + proposal_name + ', ' + current_name + ', ' + \
               repr(self.gaussian_scale) + ')'