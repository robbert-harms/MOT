"""The interfaces needed for models.

Since a lot of information about a model is needed to be able to optimize or sample it, we encapsulate all that
information in an interface. Only objects that successful implement the interfaces in this module can be optimized or
sampled using one of the optimization or sampling routines in MOT.

These interfaces expose data and modeling code. The data is represented as numpy arrays and the CL code as strings.
"""

__author__ = 'Robbert Harms'
__date__ = "2014-03-14"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class OptimizeModelInterface(object):

    @property
    def name(self):
        """Get the name of this model. This should be overwritten by the implementing model.

        Returns:
            str: A string with the name of this model.
        """
        raise NotImplementedError()

    @property
    def double_precision(self):
        """Flag to signal if we should use the double float type during calculations.

        By default we ask the cl routines to use the single precision float type, you can overwrite this with
        your own flags.

        Returns:
            boolean: if we would like to use double precision floating point during the calculations
        """
        raise NotImplementedError()

    def get_kernel_data(self):
        """Return a list of data arrays we need to load into the kernel.

        Returns:
            dict[str: mot.utils.KernelInputData]: the list of data objects we need to load in the kernel
        """
        raise NotImplementedError()

    def get_nmr_problems(self):
        """Get the number of problems we need to analyze.

        Returns:
            int: A single integer specifying the number of problem instances
        """
        raise NotImplementedError()

    def get_nmr_inst_per_problem(self):
        """Get the number of instances/data points per problem.

        The minimum is one instance per problem.

        This number represents the number of data points

        Returns:
            int: the number of instances per problem.
        """
        raise NotImplementedError()

    def get_nmr_estimable_parameters(self):
        """Get the number of estimable parameters.

        Returns:
            int: the number of estimable parameters
        """
        raise NotImplementedError()

    def get_pre_eval_parameter_modifier(self):
        """Return code that needs to be run prior to model evaluation or objective function calculation.

        This is meant to contain possible parameter transformations that need to be executed only once for a
        given set of parameters. Having this in a separate function gives a speed gain.

        Model optimization routines need to be aware that they need to call this function prior to calling any of:

        * :meth:`~get_model_eval_function`
        * :meth:`~get_objective_per_observation_function`


        Returns:
            mot.utils.NamedCLFunction: a named CL function with the following signature:

                .. code-block:: c

                    void <func_name>(mot_data_struct* data, mot_float_type* x);

                Changes may happen in place in the ``x`` parameter.
        """
        raise NotImplementedError()

    def get_model_eval_function(self):
        """Get the evaluation function that evaluates the model at the given parameters.

        The returned function should not do any error calculations,
        it should merely return the result of evaluating the model for the given parameters.

        Returns:
            mot.utils.NamedCLFunction: a named CL function with the following signature:

                .. code-block:: c

                    double <func_name>(mot_data_struct* data, const mot_float_type* const x, uint observation_index);
        """
        raise NotImplementedError()

    def get_objective_per_observation_function(self):
        """Get the objective function that returns the objective value at a measurement instance.

        This should return the objective values (of each instance point) as such that when linearly summed, we have the
        complete objective value.

        Returns:
            mot.utils.NamedCLFunction: A CL function with signature:

                .. code-block:: c

                    double <func_name>(mot_data_struct* data, const mot_float_type* const x, uint observation_index);
        """
        raise NotImplementedError()

    def get_initial_parameters(self):
        """Get a two dimensional matrix with the initial parameters (starting points) for every voxel.

        Returns:
            ndarray: A two dimensional matrix with on the first axis the problem instances and on the second
                the parameter values per problem instance
        """
        raise NotImplementedError()

    def get_lower_bounds(self):
        """Get for each estimable parameter the lower bounds.

        Returns:
            list: For every estimable parameter a scalar or vector with the the lower bound(s) for that parameter.
                For infinity use np.inf.
        """
        raise NotImplementedError()

    def get_upper_bounds(self):
        """Get for each estimable parameter the upper bounds.

        Returns:
            list: For every estimable parameter a scalar or vector with the the upper bound(s) for that parameter.
                For infinity use np.inf.
        """
        raise NotImplementedError()

    def finalize_optimized_parameters(self, parameters):
        """Called by the optimization routine after optimization to finalize the optimized parameters.

        This can be used to do some automatic post-processing similar to for example what the
        ``get_pre_eval_parameter_modifier`` can do within the kernel.

        Args:
            parameters (ndarray): the set of parameters after optimization. This method may change this array
                in place.

        Returns:
            ndarray: the updated parameters. While changes may be done in place, one must return the parameters
                one would like to use.
        """
        raise NotImplementedError()


class SampleModelInterface(OptimizeModelInterface):
    """Extends the OptimizeModelInterface with information for sampling purposes.

    This specific interface is tied to sampling with the Metropolis Hastings Random Walk sampler as
    implement in :class:`mot.cl_routines.sampling.metropolis_hastings.MetropolisHastings`.

    To be able to sample a model we (in principle) need to have:

    * a log likelihood function;
    * a proposal function;
    * and a prior function


    Proposal functions can be symmetric (if it holds that ``q(x|x') == q(x'|x)``) or
    non symmetric (i.e. ``q(x|x') != q(x'|x)``). In the case of non-symmetric proposals we need to
    have a function to get the probability log likelihood of the proposal.
    This indicates the need for two more pieces of information:

    * test if the proposal is symmetric
    * proposal log PDF function


    A trick in sampling is to have auto-adapting proposals. These proposals commonly have a distribution with a
    standard deviation that varies in time. The idea is that if the distribution is too tight (low std) only a few
    of the proposed samples are accepted and we need to broaden the distribution (increase the std). On the other
    hand, if the std is too high the jumps might not get accepted. This leads us to the following
    additional functionality:

    * proposal state update function


    Since OpenCL < 2.1 does not allow state variables in functions and also does not support classes, we need to find
    a way to store the state of the proposal distribution inside the kernel function. For that, each proposal
    CL function has as additional parameter the ``proposal_state``. The initial state can be obtained from
    this class and needs to be handed to the proposal functions in the kernels.

    Finally, this interface requires to you specify a :class:`mot.cl_routines.sampling.metropolis_hastings.MHState`
    that specifies the current state of the sampler. This can be set to a default state when starting sampling
    or to the output of a previous run to continue sampling.
    """

    def get_proposal_state(self):
        """Get for every problem instance the list of parameter values to use in the the adaptable proposal.

        Returns:
            ndarray: per problem instance the proposal parameter values that are adaptable.
        """
        raise NotImplementedError()

    def get_log_likelihood_per_observation_function(self):
        """Get the (complete) CL Log Likelihood function that evaluates the given instance under a noise model.

        This should return the LL's such that when linearly summed they yield the total log likelihood of the model.

        Returns:
            mot.utils.NamedCLFunction: A function of the kind:
                .. code-block:: c

                    double <fname>(mot_data_struct* data, const mot_float_type* const x, uint observation_index);
        """
        raise NotImplementedError()

    def is_proposal_symmetric(self):
        """Check if the entire proposal distribution is symmetric: ``q(x|x') == q(x'|x)``.

        Returns:
            boolean: True if the proposal distribution is symmetric, false otherwise.
        """
        raise NotImplementedError()

    def get_proposal_logpdf(self, address_space_proposal_state='private'):
        """Get the probability density function of the proposal in log space (as a CL string).

        This density function is used if the proposal is not symmetric.

        Args:
            address_space_proposal_state (str): the CL address space of the proposal state vector.
                Defaults to ``private``.

        Returns:
            mot.utils.NamedCLFunction: A function with the signature:

                .. code-block:: c

                    double <func_name>(uint param_ind, mot_float_type proposal,
                                       mot_float_type current,
                                       <address_space_proposal_state> mot_float_type* const proposal_state);


            Where ``param_ind`` is the index of the parameter we would like to get the proposal from,
            ``current`` is the current value of that parameter and ``proposal`` the proposal value of the parameter.
            The final argument ``proposal_state`` are the current settings of the proposal function.

            It should return for the requested parameter a value ``q(proposal | current)``, the log Probability
            Density Function (log PDF) of the proposal given the current value.
        """
        raise NotImplementedError()

    def get_proposal_function(self, address_space_proposal_state='private'):
        """Get a proposal function that returns proposals for a requested parameter.

        Args:
            address_space_proposal_state (str): the CL address space of the proposal state vector.
                Defaults to ``private``.

        Returns:
            mot.utils.NamedCLFunction: A function with the signature:

                .. code-block:: c

                    mot_float_type <func_name>(
                        uint param_ind,
                        mot_float_type current,
                        void* rng_data,
                        <address_space_proposal_state> mot_float_type* const proposal_state);


            Where ``param_ind`` is the index of the parameter for which we want the proposal and
            ``current`` is the current value of that parameter. The argument ``proposal_state`` is the
            state of the proposal distribution. One can obtain random numbers with:
            .. code-block:: c

                float randomnr = frand(rng_data);
        """
        raise NotImplementedError()

    def get_proposal_state_update_function(self, address_space='private'):
        """Get the function to update the proposal parameters

        Args:
            address_space (str): the address space of (all) the given arguments, defaults to ``private``

        Returns:
            mot.utils.NamedCLFunction: A function with the signature:
                .. code-block:: c

                    void <func_name>(<address_space> mot_float_type* const proposal_state,
                                     <address_space> ulong* const sampling_counter,
                                     <address_space> ulong* const acceptance_counter);

                The ``proposal_state`` holds the current value of all the adaptable proposal parameters and is
                of length equal to the number of adaptable parameters. The ``sampling_counter`` holds the number of
                samples drawn since last update (per parameter) and ``acceptance_counter`` holds the number of samples
                that where accepted since the last update. Both are of length equal to the total number of parameters
                in the model (!). The implementing function is free to overwrite the values in each array.
        """
        raise NotImplementedError()

    def proposal_state_update_uses_variance(self):
        """Check if the proposal state update function requires the variance of each of the parameters.

        If none of the proposal update functions require the parameter variance then we can save memory in the
        kernel by not calculating them.

        Returns:
            boolean: if at least one parameter proposal state update function requires the parameter variance
                return True, else return False.
        """
        raise NotImplementedError()

    def get_log_prior_function(self, address_space_parameter_vector='private'):
        """Get the prior function that returns the prior information about the given parameters.

        The prior function must be in log space.

        Args:
            address_space_parameter_vector (str): the address space to use for the parameter vector
                by default this is set to ``private``.

        Returns:
            mot.utils.NamedCLFunction: A function with the signature:
                .. code-block:: c

                    mot_float_type <func_name>(
                        mot_data_struct* data,
                        <address_space_parameter_vector> const mot_float_type* const x
                    );

            Which is called by the sampling routine to calculate the posterior probability.
        """
        raise NotImplementedError()

    def get_metropolis_hastings_state(self):
        """Get the current state of the Metropolis Hastings sampler.

        This can be used to continue execution of an MH sampling from a previous point in time.

        Returns:
            mot.cl_routines.sampling.metropolis_hastings.MHState: the current Metropolis Hastings state
        """
        raise NotImplementedError()


class NumericalDerivativeInterface(OptimizeModelInterface):
    """Extends the model with information necessary for calculating numerical derivatives of the objective function.

    For calculating derivatives (gradients / Hessians) numerically, we need a likelihood function and some additional
    information, like the step size for each parameter, a method for checking boundary conditions and possible parameter
    transformations for circular parameters. All these extra elements are represented in this interface.
    """

    def numdiff_get_max_step(self):
        """Get for each estimable parameter the maximum step size to use for calculating numerical derivatives.

        The derivative calculation method typically uses an adaptive step size to determine the step with the best
        trade-off between numerical errors and localization of the derivative. This method must return the
        initial and largest step size to use.

        Returns:
            list[float]: per parameter a single float with the step size for that parameter
        """
        raise NotImplementedError()

    def numdiff_get_scaling_factors(self):
        """Get for each estimable parameter a scaling factor that is to be used for scaling this parameter to unitary.

        Since numerical differentiation is sensitive to differences in step sizes, it is better to rescale
        the parameters to a unitary range instead of changing the step sizes for the parameters.

        This should return numbers such that when the parameter is multiplied with this value, the magnitude of the
        parameter is about one.

        Returns:
            list[float]: per estimable parameter a single float with the parameter scaling to use for that parameter.
                Use 1 as identity.
        """
        raise NotImplementedError()

    def numdiff_use_bounds(self):
        """Check for each parameter if we should be using the bounds for that parameter when taking the derivative.

        Returns:
            list[bool]: per parameter a boolean to identify if we should use the bounds for that parameter.
        """
        raise NotImplementedError()

    def numdiff_use_upper_bounds(self):
        """Check for each parameter if we should be using the upper bounds when taking the derivative.

        This is only used if use_bounds is True for a parameter.

        Returns:
            list[bool]: per parameter a boolean to identify if we should use the upper bounds for that parameter.
        """
        raise NotImplementedError()

    def numdiff_use_lower_bounds(self):
        """Check for each parameter if we should be using the lower bounds when taking the derivative.

        This is only used if use_bounds is True for a parameter.

        Returns:
            list[bool]: per parameter a boolean to identify if we should use the lower bounds for that parameter.
        """
        raise NotImplementedError()

    def numdiff_parameter_transformation(self):
        """A transformation that can prepare the parameter +/- the proposed step for evaluation in the model function.

        Some parameters require for example a modulus operation before the proposed step can be used in the model
        This is the place to define them. Please note that this function does not need to incorporate boundary checks
        as those are handled already by the numerical differentiation routine.

        Returns:
            mot.utils.NamedCLFunction: A function with the signature:
                .. code-block:: c

                    void <func_name>(mot_data_struct* data, mot_float_type* params);

                Where the data is the kernel data struct and params is the vector with the suggested parameters and
                which can be modified in place. Note that this is called two times, one with the parameters plus
                the step and one time without.
        """
        raise NotImplementedError()
