from mot.configuration import get_default_proposal_update


__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ParameterProposal(object):
    """The parameter proposals are meant for use during sampling.

    Proposal functions are an essential part in sampling as they provide new candidate points for the sampler.

    The proposals allow for dynamic updates to better adapt to the landscape during sampling.
    """

    def is_symmetric(self):
        """Check if this proposal is symmetric. That is, if q(x|y) == q(y|x).

        Returns:
            boolean: if the proposal function is symmetric return True, else False.
        """
        raise NotImplementedError()

    def is_adaptable(self):
        """Check if this proposal is adaptable, i.e., if we need to update any of its parameters during sampling.

        Returns:
            boolean: return True if the proposal is adaptable, else False
        """
        raise NotImplementedError()

    def get_parameters(self):
        """The proposal parameters.

        Returns:
            list of ProposalParameter: the proposal parameter objects used by this proposal
        """
        raise NotImplementedError()

    def get_proposal_function(self):
        """Get the proposal function as a CL string. This should include include guards (#ifdef's).

        This should follow the signature:

        .. code-block: c

            mot_float_type <proposal_fname>(mot_float_type current, void* rng_data, <additional_parameters>)

        That is, it can have more than two parameter, but the first two are obligatory. The additional parameters
        are given by :meth:`get_parameters`.

        Returns:
            str: The cl function
        """
        raise NotImplementedError()

    def get_proposal_function_name(self):
        """Get the name of the proposal function call.

         This is used by the model builder to construct the call to the proposal function.

         Returns:
            str: name of the function
        """
        raise NotImplementedError()

    def get_proposal_logpdf_function(self):
        """Get the proposal pdf function as a CL string.

        This function is used if the proposal is not symmetric. The implementation should include include guards.

        This should follow the signature:

        .. code-block: c

            mot_float_type <proposal_pdf_fname>(mot_float_type proposal, mot_float_type current,
                                                <additional_parameters>)

        Returns:
            str: The proposal log pdf function as a CL string
        """
        raise NotImplementedError()

    def get_proposal_logpdf_function_name(self):
        """Get the name of the proposal logpdf function call.

         This is used by the model builder to construct the call to the proposal logpdf function.

         Returns:
            str: name of the function
        """
        raise NotImplementedError()

    def get_proposal_update_function(self):
        """Get the proposal update function to use for updating the adaptable parameters.

        Returns:
            mot.model_building.parameter_functions.proposal_updates.ProposalUpdate: the proposal update function
                defining the update mechanism
        """
        raise NotImplementedError()


class ProposalParameter(object):

    def __init__(self, name, default_value, adaptable):
        """Container class for parameters of a proposal function.

        Args:
            default_value (double): the parameter value
            adaptable (boolean): if this parameter is adaptable during sampling

        Attributes:
            default_value (double): the parameter value
            adaptable (boolean): if this parameter is adaptable
        """
        self.name = name
        self.default_value = default_value
        self.adaptable = adaptable


class SimpleProposal(ParameterProposal):

    def __init__(self, proposal_body, proposal_name, parameters, is_symmetric=True, logpdf_body='return 0;',
                 proposal_update_function=None):
        """Simple proposal template class.

        By default this assumes that the proposal you are generating is symmetric. If so, the proposal logpdf function
        can be reduced to a scalar since calculating it is unnecessary.

        Args:
            proposal_body (str): the body of the proposal code. Environment variables are ``current`` for the current
                position of this parameter and ``rng_data`` that can be used to generate random numbers.
            proposal_name (str): the name of this proposal
            parameters (list): the list of :class:`ProposalParameters` used by this proposal
            is_symmetric (boolean): if this proposal is symmetric, that is, if ``q(x|y) == q(y|x)``.
            logpdf_body (str): if the proposal is not symmetric we need a PDF function to calculate the probability of
                ``q(x|y)`` and ``q(y|x)``. It should return the log of the probability. If the proposal is symmetric
                this parameter need not be specified and defaults to returning a scalar.
            proposal_update_function (mot.model_building.parameter_functions.proposal_updates.ProposalUpdate): the
                proposal update function to use. For the default check the mot configuration.
        """
        self._proposal_body = proposal_body
        self._proposal_name = proposal_name
        self._parameters = parameters
        self._is_symmetric = is_symmetric
        self._logpdf_body = logpdf_body
        self._proposal_update_function = proposal_update_function or get_default_proposal_update()

    def is_symmetric(self):
        return self._is_symmetric

    def is_adaptable(self):
        return any(p.adaptable for p in self._parameters)

    def get_parameters(self):
        return self._parameters

    def get_proposal_function(self):
        params = ['mot_float_type current', 'void* rng_data']
        params.extend('mot_float_type {}'.format(p.name) for p in self._parameters)

        return '''
            #ifndef {include_guard_name}
            #define {include_guard_name}

            mot_float_type {function_name}({params}){{
                {function_body}
            }}

            #endif //{include_guard_name}
        '''.format(include_guard_name='PROPOSAL_{}'.format(self._proposal_name.upper()),
                   function_name=self.get_proposal_function_name(),
                   params=', '.join(params),
                   function_body=self._proposal_body)

    def get_proposal_function_name(self):
        return 'proposal_{}'.format(self._proposal_name)

    def get_proposal_logpdf_function(self):
        params = ['mot_float_type current', 'mot_float_type other']
        params.extend('mot_float_type {}'.format(p.name) for p in self._parameters)

        return '''
            #ifndef {include_guard_name}
            #define {include_guard_name}

            mot_float_type {function_name}({params}){{
                {function_body}
            }}

            #endif //{include_guard_name}
        '''.format(include_guard_name='PROPOSAL_LOGPDF_{}'.format(self._proposal_name.upper()),
                   function_name=self.get_proposal_logpdf_function_name(),
                   params=', '.join(params),
                   function_body=self._logpdf_body)

    def get_proposal_logpdf_function_name(self):
        return 'proposal_logpdf_{}'.format(self._proposal_name)

    def get_proposal_update_function(self):
        return self._proposal_update_function


class GaussianProposal(SimpleProposal):

    def __init__(self, std=1.0, adaptable=True, proposal_update_function=None):
        """Create a new proposal function using a Gaussian distribution with the given scale.

        Args:
            std (float): The scale of the Gaussian distribution.
            adaptable (boolean): If this proposal is adaptable during sampling
            proposal_update_function (mot.model_building.parameter_functions.proposal_updates.ProposalUpdate): the
                proposal update function to use. Defaults to the one in the current mot configuration.
        """
        parameters = [ProposalParameter('std', std, adaptable)]
        super(GaussianProposal, self).__init__(
            'return fma(std, (mot_float_type)frandn(rng_data), current);',
            'gaussian',
            parameters,
            proposal_update_function=proposal_update_function
        )


class CircularGaussianProposal(SimpleProposal):

    def __init__(self, modulus, std=1.0, adaptable=True, proposal_update_function=None):
        """A Gaussian distribution which loops around the given modulus.

        Args:
            modulus (float): at which point we loop around
            std (float): The scale of the Gaussian distribution.
            adaptable (boolean): If this proposal is adaptable during sampling
            proposal_update_function (mot.model_building.parameter_functions.proposal_updates.ProposalUpdate): the
                proposal update function to use. Defaults to the one in the current mot configuration.
        """
        parameters = [ProposalParameter('std', std, adaptable)]
        super(CircularGaussianProposal, self).__init__(
            '''
                double x1 = fma(std, (mot_float_type)frandn(rng_data), current);
                double x2 = {};
                return (mot_float_type) (x1 - floor(x1 / x2) * x2);
            '''.format(modulus),
            'circular_gaussian',
            parameters,
            proposal_update_function=proposal_update_function
        )
