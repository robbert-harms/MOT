__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractParameterProposal(object):
    """The parameter proposals are meant for use during sampling.

    Proposal functions are an essential part in sampling as they provide new candidate points for the sampler.

    The proposals allow for dynamic updates to better adapt to the landscape during sampling.
    """

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

        .. code-block: c

            mot_float_type <proposal_fname>(mot_float_type current, void* rng_data, <additional_parameters>)

        That is, it can have more than two parameter, but the first two are obligatory. The additional parameters
        are given by :meth:`get_parameters`.

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
        """Get the proposal pdf function as a CL string.

        This function is used if the proposal is not symmetric. The implementation should include include guards.

        This should follow the signature:

        .. code-block: c

            mot_float_type <proposal_pdf_fname>(mot_float_type proposal, mot_float_type current,
                                                <additional_parameters>)

        Returns:
            str: The proposal log pdf function as a CL string
        """

    def get_proposal_logpdf_function_name(self):
        """Get the name of the proposal logpdf function call.

         This is used by the model builder to construct the call to the proposal logpdf function.

         Returns:
            str: name of the function
        """

    def get_proposal_update_function(self, address_space='private'):
        """Get the proposal update function to use for updating the adaptable parameters.

        Args:
            address_space (str): the address space for the arguments passed to the update function

        Returns:
            str: the proposal update function to use
        """

    def get_proposal_update_function_name(self):
        """Get the name of the proposal update function

        Returns:
            str: the name of the proposal update function
        """


class ProposalUpdate(object):

    def get_update_function(self, proposal_parameters, address_space='private'):
        """Get the update function to update the proposal parameters of one of the proposals.

        Args:
            proposal_parameters (list of ProposalParameter): the list of proposal parameters to (possibly use) in the
                update function. It will only use the parameters that have ``adaptable`` set to True.
            address_space (str): the address space to use for the function parameters
        """

    def get_function_name(self, proposal_parameters):
        """Get the name of the proposal update function.

        Args:
            proposal_parameters (list of ProposalParameter): the list of proposal parameters to (possibly use) in the
                update function. It will only use the parameters that have ``adaptable`` set to True.

        Returns:
            str: the name of the function returned by :meth:`get_update_function`
        """


class SimpleProposalUpdate(ProposalUpdate):

    def __init__(self, function_name):
        """A simple proposal update function template.

        Args:
            function_name (str): the name of this proposal update function, try to choose an unique name.

        """
        self._function_name = function_name

    def get_update_function(self, proposal_parameters, address_space='private'):
        return self._update_function_template('', proposal_parameters, address_space)

    def _update_function_template(self, function_body, proposal_parameters, address_space):
        params = ['{address_space} mot_float_type* const {name}'.format(address_space=address_space, name=p.name)
                  for p in proposal_parameters if p.adaptable]

        params.extend(['{address_space} uint* const sampling_counter'.format(address_space=address_space),
                       '{address_space} uint* const acceptance_counter'.format(address_space=address_space)])

        return '''
            #ifndef {include_guard_name}
            #define {include_guard_name}

            mot_float_type {function_name}({params}){{
                {function_body}
            }}

            #endif //{include_guard_name}

        '''.format(include_guard_name='PROPOSAL_UPDATE_{}'.format(self._function_name.upper()),
                   function_name=self.get_function_name(proposal_parameters), params=', '.join(params),
                   function_body=function_body)

    def get_function_name(self, proposal_parameters):
        return 'proposal_update_{}_{}'.format(self._function_name,
                                              len([True for p in proposal_parameters if p.adaptable]))


class NoOptUpdateFunction(SimpleProposalUpdate):

    def __init__(self):
        """This is the no-operation update function. It does not update the proposal parameters."""
        super(NoOptUpdateFunction, self).__init__('no_opt')

    def get_update_function(self, proposal_parameters, address_space='private'):
        return self._update_function_template('', proposal_parameters, address_space)


class AcceptanceRateScaling(SimpleProposalUpdate):

    def __init__(self, target_acceptance_rate=0.44, batch_size=50, damping_factor=500):
        """Scales the proposal parameter (typically the std) such that it oscillates towards the chosen acceptance rate.

        This uses an scaling similar to the one in: "Examples of Adaptive MCMC",
        Gareth O. Roberts & Jeffrey S. Rosenthal (2009)

        This class implements the delta function as: :math:`\delta(n) = \sqrt{1 / (d*n)}`.
        Where n is the current batch index and d is the damping factor.

        With the default damping factor of 500, delta reaches a scaling of 0.01 in 20 batches. At a default batch size
        of 50 that amounts to 1000 samples.

        Args:
            target_acceptance_rate (float): the target acceptance rate between 0 and 1.
            batch_size (int): the size of the batches inbetween which we update the parameters
        """
        super(AcceptanceRateScaling, self).__init__('acceptance_rate_scaling')
        self._target_acceptance_rate = target_acceptance_rate
        self._batch_size = batch_size

        if target_acceptance_rate > 1 or target_acceptance_rate < 0:
            raise ValueError('The target acceptance rate should be '
                             'within [0, 1], {} given.'.format(target_acceptance_rate))

    def get_update_function(self, proposal_parameters, address_space='private'):
        body = '''
            if(*sampling_counter % {batch_size} == 0){{

                mot_float_type delta = sqrt(1.0/(400 * (*sampling_counter / {batch_size})));

                if(*acceptance_counter / (*sampling_counter % {batch_size}) > {target_ar}){{
                    *std *= exp(delta);
                }}
                else{{
                    *std /= exp(delta);
                }}

                *acceptance_counter = 0;
            }}
        '''.format(batch_size=self._batch_size, target_ar=self._target_acceptance_rate)
        return self._update_function_template(body, proposal_parameters, address_space)


class FSLAcceptanceRateScaling(SimpleProposalUpdate):

    def __init__(self, batch_size=50):
        """An acceptance rate scaling algorithm found in a Neuroscience package called FSL.

        This scaling algorithm scales the std. by :math:`\sqrt(a/(n - a))` where a is the number of accepted samples
        in the last batch and n is the batch size. Its goal is to balance the acceptance rate at 0.5.

        So far, the author of this function in MOT has not been able to find theoretical support for this scaling
        algorithm. Please use this heuristic with caution.

        Args:
            batch_size (int): the size of the batches inbetween which we update the parameters
        """
        super(FSLAcceptanceRateScaling, self).__init__('fsl_acceptance_rate_scaling')
        self._batch_size = batch_size

    def get_update_function(self, proposal_parameters, address_space='private'):
        body = '''
            if(*sampling_counter == {batch_size}){{
                if(*sampling_counter != *acceptance_counter){{
                    *std = min(*std * sqrt((mot_float_type)*acceptance_counter /
                                            (*sampling_counter - *acceptance_counter)), (mot_float_type)1e10);
                }}

                *sampling_counter = 0;
                *acceptance_counter = 0;
            }}
        '''.format(batch_size=self._batch_size)
        return self._update_function_template(body, proposal_parameters, address_space)


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


class SimpleProposal(AbstractParameterProposal):

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
            proposal_update_function (ProposalUpdate): the proposal update function to use. Defaults
                to :class:`AcceptanceRateScaling`.
        """
        self._proposal_body = proposal_body
        self._proposal_name = proposal_name
        self._parameters = parameters
        self._is_symmetric = is_symmetric
        self._logpdf_body = logpdf_body
        self._proposal_update_function = proposal_update_function or AcceptanceRateScaling()

    def is_symmetric(self):
        return self._is_symmetric

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

    def get_proposal_update_function(self, address_space='private'):
        return self._proposal_update_function.get_update_function(self._parameters, address_space)

    def get_proposal_update_function_name(self):
        return self._proposal_update_function.get_function_name(self._parameters)


class GaussianProposal(SimpleProposal):

    def __init__(self, std=1.0, adaptable=True, proposal_update_function=None):
        """Create a new proposal function using a Gaussian distribution with the given scale.

        Args:
            std (float): The scale of the Gaussian distribution.
            adaptable (boolean): If this proposal is adaptable during sampling
            proposal_update_function (ProposalUpdate): the proposal update function to use. Defaults
                to :class:`AcceptanceRateScaling`.
        """
        parameters = [ProposalParameter('std', std, adaptable)]
        super(GaussianProposal, self).__init__(
            'return fma(std, (mot_float_type)frandn(rng_data), current);',
            'gaussian',
            parameters,
            proposal_update_function=proposal_update_function or FSLAcceptanceRateScaling()
        )


class CircularGaussianProposal(SimpleProposal):

    def __init__(self, modulus, std=1.0, adaptable=True, proposal_update_function=None):
        """A Gaussian distribution which loops around the given modulus.

        Args:
            modulus (float): at which point we loop around
            std (float): The scale of the Gaussian distribution.
            adaptable (boolean): If this proposal is adaptable during sampling
            proposal_update_function (ProposalUpdate): the proposal update function to use. Defaults
                to :class:`FSLAcceptanceRateScaling`.
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
            proposal_update_function=proposal_update_function or FSLAcceptanceRateScaling()
        )


class ClippedGaussianProposal(SimpleProposal):

    def __init__(self, std=1.0, adaptable=True, min_val=0, max_val=1, proposal_update_function=None):
        """Create a new proposal function using a Gaussian distribution with the given scale.

        Args:
            std (float): The scale of the Gaussian distribution.
            adaptable (boolean): If this proposal is adaptable during sampling
            min_val (float): the minimum value allowed, everything above is clipped to this value
            max_val (float): the maximum value allowed, everything above is clipped to this value
            proposal_update_function (ProposalUpdate): the proposal update function to use. Defaults
                to :class:`FSLAcceptanceRateScaling`.
        """
        parameters = [ProposalParameter('std', std, adaptable)]
        super(ClippedGaussianProposal, self).__init__(
            '''
                return clamp(std * frandn(rng_data) + current,
                             (mot_float_type){},
                             (mot_float_type){});
            '''.format(min_val, max_val),
            'clipped_gaussian',
            parameters,
            proposal_update_function=proposal_update_function or FSLAcceptanceRateScaling()
        )
