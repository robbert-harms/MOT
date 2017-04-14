__author__ = 'Robbert Harms'
__date__ = "2017-03-02"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ProposalUpdate(object):

    def get_update_function(self, proposal_parameters, address_space='private'):
        """Get the update function to update the proposal parameters of one of the proposals.

        Args:
            proposal_parameters (list of ProposalParameter): the list of proposal parameters to (possibly use) in the
                update function. It will only use the parameters that have ``adaptable`` set to True.
            address_space (str): the address space to use for the function parameters
        """
        raise NotImplementedError()

    def get_function_name(self, proposal_parameters):
        """Get the name of the proposal update function.

        Args:
            proposal_parameters (list of ProposalParameter): the list of proposal parameters to (possibly use) in the
                update function. It will only use the parameters that have ``adaptable`` set to True.

        Returns:
            str: the name of the function returned by :meth:`get_update_function`
        """
        raise NotImplementedError()

    def uses_parameter_variance(self):
        """Check if this proposal update function uses the parameter variance.

        If not, we will not provide it, this saves memory in the kernel.

        Returns:
            boolean: if this proposal update function uses the parameter variance
        """
        raise NotImplementedError()

    def uses_jump_counters(self):
        """Check if this proposal update function uses the jump counters (jump counter and acceptance counter).

        If not, we will not provide it, this saves memory in the kernel.

        Returns:
            boolean: if this proposal update function uses the jump counters
        """
        raise NotImplementedError()


class SimpleProposalUpdate(ProposalUpdate):

    def __init__(self, function_name, uses_parameter_variance=False, uses_jump_counters=True):
        """A simple proposal update function template.

        Args:
            function_name (str): the name of this proposal update function, try to choose an unique name.
            uses_parameter_variance (boolean): if this proposal requires the parameter variance. Enable if you need it
                 in your update function.
            uses_jump_counters (boolean): if this proposal uses jump counters for its workings. This is enabled
                by default. You can disable it for speed.
        """
        self._function_name = function_name
        self._uses_parameter_variance = uses_parameter_variance
        self._uses_jump_counters = uses_jump_counters

    def get_update_function(self, proposal_parameters, address_space='private'):
        return self._update_function_template('', proposal_parameters, address_space)

    def _update_function_template(self, function_body, proposal_parameters, address_space):
        params = ['{address_space} mot_float_type* const {name}'.format(address_space=address_space, name=p.name)
                  for p in proposal_parameters if p.adaptable]

        if self.uses_jump_counters():
            params.extend(['{address_space} ulong* const sampling_counter'.format(address_space=address_space),
                           '{address_space} ulong* const acceptance_counter'.format(address_space=address_space)])

        if self.uses_parameter_variance():
            params.append('mot_float_type parameter_variance'.format(address_space=address_space))

        return '''
            #ifndef {include_guard_name}
            #define {include_guard_name}

            void {function_name}({params}){{
                {function_body}
            }}

            #endif //{include_guard_name}

        '''.format(include_guard_name='PROPOSAL_UPDATE_{}'.format(self._function_name.upper()),
                   function_name=self.get_function_name(proposal_parameters), params=', '.join(params),
                   function_body=function_body)

    def get_function_name(self, proposal_parameters):
        return 'proposal_update_{}_{}'.format(self._function_name,
                                              len([True for p in proposal_parameters if p.adaptable]))

    def uses_parameter_variance(self):
        return self._uses_parameter_variance

    def uses_jump_counters(self):
        return self._uses_jump_counters


class NoOperationUpdateFunction(SimpleProposalUpdate):

    def __init__(self):
        """This is the no-operation update function. It does not update the proposal parameters."""
        super(NoOperationUpdateFunction, self).__init__('no_opt')

    def get_update_function(self, proposal_parameters, address_space='private'):
        return self._update_function_template('', proposal_parameters, address_space)


class AcceptanceRateScaling(SimpleProposalUpdate):

    def __init__(self, target_acceptance_rate=0.44, batch_size=50, damping_factor=1):
        """Scales the proposal parameter (typically the std) such that it oscillates towards the chosen acceptance rate.

        This uses an scaling similar to the one in: "Examples of Adaptive MCMC",
        Gareth O. Roberts & Jeffrey S. Rosenthal (2009)

        This class implements the delta function as: :math:`\delta(n) = \sqrt{1 / (d*n)}`.
        Where n is the current batch index and d is the damping factor.

        As an example, with a damping factor of 500, delta reaches a scaling of 0.01 in 20 batches. At a batch size
        of 50 that would amount to 1000 samples.

        Args:
            target_acceptance_rate (float): the target acceptance rate between 0 and 1.
            batch_size (int): the size of the batches inbetween which we update the parameters
        """
        super(AcceptanceRateScaling, self).__init__('acceptance_rate_scaling')
        self._target_acceptance_rate = target_acceptance_rate
        self._batch_size = batch_size
        self._damping_factor = damping_factor

        if target_acceptance_rate > 1 or target_acceptance_rate < 0:
            raise ValueError('The target acceptance rate should be '
                             'within [0, 1], {} given.'.format(target_acceptance_rate))

    def get_update_function(self, proposal_parameters, address_space='private'):
        body = '''
            if(*sampling_counter % {batch_size} == 0){{

                mot_float_type delta = sqrt(1.0/({damping_factor} * (*sampling_counter / {batch_size})));

                if(*acceptance_counter / (mot_float_type){batch_size} > {target_ar}){{
                    *std *= exp(delta);
                }}
                else{{
                    *std /= exp(delta);
                }}

                *std = clamp(*std, (mot_float_type)1e-13, (mot_float_type)1e3);

                *acceptance_counter = 0;
            }}
        '''.format(batch_size=self._batch_size, target_ar=self._target_acceptance_rate,
                   damping_factor=self._damping_factor)
        return self._update_function_template(body, proposal_parameters, address_space)


class FSLAcceptanceRateScaling(SimpleProposalUpdate):

    def __init__(self, batch_size=50, min_val=1e-13, max_val=1e3):
        """An acceptance rate scaling algorithm found in a Neuroscience package called FSL.

        This scaling algorithm scales the std. by :math:`\sqrt(a/(n - a))` where a is the number of accepted samples
        in the last batch and n is the batch size. Its goal is to balance the acceptance rate at 0.5.

        So far, the author of this function in MOT has not been able to find theoretical support for this scaling
        algorithm. Please use this heuristic with caution.

        To prevent runaway proposal values we clamp the updated parameter value between a minimum and maximum value
        specified in the constructor.

        Args:
            batch_size (int): the size of the batches in between which we update the parameters
            min_val (float): the minimum value the parameter can take
            max_val (float): the maximum value the parameter can take
        """
        super(FSLAcceptanceRateScaling, self).__init__('fsl_acceptance_rate_scaling')
        self._batch_size = batch_size
        self._min_val = min_val
        self._max_val = max_val

    def get_update_function(self, proposal_parameters, address_space='private'):
        body = '''
            if(*sampling_counter == {batch_size}){{
                *std = clamp(*std * sqrt((mot_float_type)(*acceptance_counter + 1) /
                                         ({batch_size} - *acceptance_counter + 1)),
                             (mot_float_type){min_val},
                             (mot_float_type){max_val});

                *sampling_counter = 0;
                *acceptance_counter = 0;
            }}
        '''.format(batch_size=self._batch_size, min_val=self._min_val, max_val=self._max_val)
        return self._update_function_template(body, proposal_parameters, address_space)


class SingleComponentAdaptiveMetropolis(SimpleProposalUpdate):

    def __init__(self, waiting_period=100, scaling_factor=2.4, epsilon=1e-20):
        """Uses the Single Component Adaptive Metropolis (SCAM) scheme to update the proposals.

        This uses an scaling described in: "Componentwise adaptation for high dimensional MCMC",
        Heikki Haario, Eero Saksman and Johanna Tamminen (2005). That is, it updates the proposal standard deviation
        using the variance of the chain's history.

        Args:
            waiting_period (int): only start updating the proposal std. after this many draws.
            scaling_factor (float): the scaling factor to use (the parameter ``s`` in the paper referenced).
            epsilon (float): small number to prevent the std. from collapsing to zero.
        """
        super(SingleComponentAdaptiveMetropolis, self).__init__('scam', uses_parameter_variance=True)
        self._waiting_period = waiting_period
        self._scaling_factor = scaling_factor
        self._epsilon = epsilon

    def get_update_function(self, proposal_parameters, address_space='private'):
        body = '''
            if(*sampling_counter > {waiting_period}){{
                *std = {scaling_factor} * sqrt(parameter_variance) + {epsilon};
            }}
        '''.format(waiting_period=self._waiting_period, scaling_factor=self._scaling_factor, epsilon=self._epsilon)
        return self._update_function_template(body, proposal_parameters, address_space)
