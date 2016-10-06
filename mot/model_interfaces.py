"""The interfaces needed for models.

Since a lot of information about a model is needed to be able to optimize or sample it we encapsulate all that
information in an interface. Only objects that successful implement the interfaces in this module can be optimized or
sampled using one of the optimization or sampling routines in MOT.

These interfaces expose data and modelling code. The data is encapsulated in :class:`~mot.data_adapters.DataAdapter`
and the code should be returned as CL strings. In the future, instead of CL strings we may require returning an
encapsulating object such that we can run the computations on multiple types of runtime environments
(CUDA, plain C, ...).
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
        return ""

    @property
    def double_precision(self):
        """Flag to signal if we should use the double float type during calculations.

        By default we ask the cl routines to use the single precision float type, you can overwrite this with
        your own flags.

        Returns:
            boolean: if we would like to use double precision floating point during the calculations
        """
        return False

    def get_problems_var_data(self):
        """Get a dict with all the data per problem.

        As an example, suppose per problem instance we have the data named 'observations'.
        We should then return something like:

        .. code-block:: python

            {'observations': SimpleDataAdapter(ndarray((<nmr_problems>, <number of items>)), ...), ...}

        Returns:
            dict: A dictionary where each element holds a SimpleDataAdapter from which to get the data
                The data in the adapter should contain for each problem (first dimension / rows) a
                number of items (second dimension / columns) that is used in the evaluation function.
        """
        raise NotImplementedError

    def get_problems_protocol_data(self):
        """Get a dict with the data that is constant for each problem, but differs per measurement.

        Returns:
            dict: A dict where each element holds a SimpleDataAdapter that is used in the evaluation function.
                In the CL kernel, each key is used as name in the cl data struct.
        """
        raise NotImplementedError

    def get_model_data(self):
        """Get a dict with all the model data. This is model data necessary for computations in the model.

        For data that is to large to inline in the kernel this data type may be useful.

        Returns:
            dict: The dict with all the model data in DataAdapters.
                The names of the keys are used in the CL cl data structs.
        """

    def get_nmr_problems(self):
        """Get the number of problems we need to analyze.

        Returns:
            int: A single integer specifying the number of problem instances
        """
        raise NotImplementedError

    def get_model_eval_function(self, func_name='evaluateModel'):
        """Get the evaluation function that evaluates the model at the given parameters.

        This returned function should not do any error calculations,
        it should merely return the result of evaluating the model for the given parameters.

        Please make sure the sign of the return value is correct given the following. The minimization
        routines may make use of this function and get_observation_return_function to build their
        own objective function. This is always done as: observation() - evaluation(). This means that if you
        want to optimize a function without observation data you need to make sure the evaluation function returns
        the answers with the right sign.

        Args:
            func_name (string): specifies the name of the function.

        Returns:
            str: An CL function with the signature:
                .. code-block:: c

                    mot_float_type <func_name>(const optimize_data* const data, const mot_float_type* const x,
                                            const int observation_index);
        """
        raise NotImplementedError

    def get_observation_return_function(self, func_name='getObservation'):
        """Get the CL function that returns the observation for the given problem.

        Args:
            func_name (string): specifies the name of the function.

        Returns:
            str: An CL function with the signature:
                .. code-block:: c

                    mot_float_type <func_name>(const optimize_data* const data, const int observation_index);
        """
        raise NotImplementedError

    def get_objective_function(self, func_name="calculateObjective"):
        """Get the objective function that evaluates the entire problem instance under a noise model.

        Args:
            func_name (string): specifies the name of the function.

        Returns:
            str: A CL function with signature:
                .. code-block:: c

                    mot_float_type <func_name>(const optimize_data* const data, mot_float_type* const x);
        """
        raise NotImplementedError

    def get_objective_list_function(self, func_name="calculateObjectiveList"):
        """Get the objective function returning a list with the objective functions per instance point.

        This function is used by some evaluation routines (like for example LevenbergMarquardt) that need
        a list of objective values (one per instance point), instead of a single objective function scalar.

        Args:
            func_name (str): the name of the function

        Returns:
            str: A CL function with signature:
                .. code-block:: c

                    mot_float_type <func_name>(const optimize_data* const data, mot_float_type* const x,
                                               mot_float_type* result);
        """
        raise NotImplementedError

    def get_initial_parameters(self, results_dict=None):
        """Get a two dimensional matrix with the initial parameters (starting points) for every voxel.

        Optionally, one may specify a list of previously calculated results which may be applicable to the model.
        If a parameter is found in the results_dict, those values are used for the initial parameters.

        Args:
            results_dict (dict): a dictionary with for every parameter name, a value per voxel which is (for example)
                the result of a previous calculation.

        Returns:
            ndarray: A two dimensional matrix with on the first axis the problem instances and on the second
                the parameter values per problem instance
        """
        raise NotImplementedError

    def get_lower_bounds(self):
        """Get for each estimable parameter the lower bounds.

        Returns:
            ndarray: An numpy row with on each column a single value, the lower bound for that parameter. This value
            can be the literal string '-inf' for infinity.
        """
        raise NotImplementedError

    def get_upper_bounds(self):
        """Get for each estimable parameter the upper bounds.

        Returns:
            ndarray: An numpy row with on each column a single value, the upper bound for that parameter. This value
            can be the literal string 'inf' for infinity.
        """
        raise NotImplementedError

    def get_optimized_param_names(self):
        """Get a list of names with the free parameter names (the parameters that are estimated by the routines).

        The function get_optimization_output_param_names() returns the names of all the parameter names,
        including fixed and static parameters. This should only return the names of the parameters
        that are actually used in the optimization.

        Returns:
            list of str: A list with the parameter names (in dot format) of all the estimated (free) parameters.
        """
        raise NotImplementedError

    def get_optimization_output_param_names(self):
        """Get a list with the names of the parameters, this is the list of keys to the titles and results.

        See get_optimized_param_names() for getting the names of the parameters that are actually being optimized.

        This should be a complete overview of all the maps returned from optimizing this model.

        Returns:
            list of str: a list with the parameter names
        """
        raise NotImplementedError

    def get_nmr_inst_per_problem(self):
        """Get the number of instances/data points per problem.

        The minimum is one instance per problem.

        This number represents the number of data points

        Returns:
            int: the number of instances per problem.
        """
        raise NotImplementedError

    def get_nmr_estimable_parameters(self):
        """Get the number of estimable parameters.

        Returns:
            int: the number of estimable parameters
        """
        raise NotImplementedError

    def get_parameter_codec(self):
        """Get the parameter codec from this model. This should be an implementation of AbstractCodec.

        Returns:
            AbstractCodec or None: An abstract codec model that holds the CL code for the codec transformations.
            This function may also return None, which indicates that no parameter codec is supposed to be used.
        """
        raise NotImplementedError

    def get_final_parameter_transformations(self, func_name='applyFinalParameterTransformations'):
        """Get the transformations that must be applied at the end of an optimization (or sampling) routine.

        These transformations must contain all parameter dependencies, that is, all transformation happening in the
        model function which do not happen in the codec must be present in this function.

        Suppose an optimization routine finds a set of parameters X to the the optimal set of parameters. In the
        evaluation function this set of parameters might have been transformed to a new set of parameters X' by the
        parameter dependencies. Since we, in the end, are interested in the set of parameters X', we have to apply
        the exact same transformations at the end of the optimization routine as happened in the evaluation function.

        Args:
            func_name (string): specifies the name of the function.

        Returns:
            str or None: Return None if this function is not used, else a function of the kind:
                .. code-block:: c

                    void <func_name>(const optimize_data* data, mot_float_type* x);

            Which is called for every voxel and must in place edit the x variable.
        """
        raise NotImplementedError

    def finalize_optimization_results(self, results_dict):
        """After optimization create the final dictionary with the result maps.

        In this location extra maps can be added to the results dictionary.

        This function behaves as a procedure and as a function. The input dict can be updated in place, but it should
        also return a dict but that is merely for the purpose of chaining.

        Args:
            results_dict (dict): A dictionary with as keys the names of the parameters and as values the 1d maps with
                for each voxel the optimized parameter value. The given dictionary can be altered by this function.

        Returns:
            dict: The same result dictionary but with updated values or with additional maps.
                It should at least return the results_dict.
        """
        return results_dict


class SampleModelInterface(OptimizeModelInterface):

    def __init__(self):
        """Extends the OptimizeModelInterface with information for sampling purposes.

        To be able to sample a model we need to have a:

        * log likelihood function
        * a proposal function
        * a prior function


        Proposal functions can be symmetric (if it holds that ``q(x|x') == q(x'|x)``) or
        non symmetric (i.e. ``q(x|x') != q(x'|x)``). In the case of non-symmetric proposals we need to
        have a function to get the probability log likelihood of the proposal.
        This indicates the need for two more pieces of information:

        * test if the proposal is symmetric
        * proposal log PDF function


        A trick in sampling is to have auto-adapting proposals. These proposals commonly have a distribution with a
        standard deviation that varies in time. The idea is that if the distribution is too tight (low std) only a few
        of the proposed samples are accepted and we need to broaden the distribution (increase the std). On the other
        hand, if the std is to high we are jumping around to much in search space. This leads us to the following
        additional functionality:

        * proposal state update function


        Since OpenCL < 2.1 does not allow for state variables in functions and does not have classes we need to find
        a way to store the state of the proposal distribution inside the kernel function. For that, each proposal
        CL function has as additional parameter the ``proposal_state``. The initial state can be obtained from
        this class and needs to be handed to the proposal functions.
        """
        super(SampleModelInterface, self).__init__()

    def get_proposal_state(self):
        """Get a list of parameter values for the adaptable proposal parameters.

        Returns:
            list: list of float with the proposal parameter values that are adaptable.
        """
        raise NotImplementedError

    def get_log_likelihood_function(self, func_name="getLogLikelihood", evaluation_model=None, full_likelihood=True):
        """Get the CL Log Likelihood function that evaluates the entire problem instance under a noise model

        Args:
            func_name (string): specifies the name of the function.
            evaluation_model (EvaluationModel): the evaluation model to use for the log likelihood. If not given
                we use the one defined in the model.
            full_likelihood (boolean): if we want the complete likelihood, or if we can drop the constant terms.
                The default is the complete likelihood. Disable for speed.

        Returns:
            str: A function of the kind:
                .. code-block:: c

                    mot_float_type <func_name>(const optimize_data* const data, mot_float_type* const x);
        """
        raise NotImplementedError

    def is_proposal_symmetric(self):
        """Check if the entire proposal distribution is symmetric: ``q(x|x') == q(x'|x)``.

        Returns:
            boolean: True if the proposal distribution is symmetric, false otherwise.
        """
        raise NotImplementedError

    def get_proposal_logpdf(self, func_name='getProposalLogPDF'):
        """Get the probability density function of the proposal in log space (as a CL string).

        This density function is used if the proposal is not symmetric.

        Args:
            func_name (str): the CL function name of the returned function

        Returns:
            str: A function with the signature:
                .. code-block:: c

                    mot_float_type <func_name>(const int i, const mot_float_type proposal,
                                               const mot_float_type current, mot_float_type* const proposal_state)

            Where ``i`` is the index of the parameter we would like to get the proposal from, ``current`` is the current
            value of that parameter and ``proposal`` the proposal value of the parameter. The final argument
            ``proposal_state`` are the current settings of the proposal function.

            It should return for the requested parameter a value ``q(proposal | current)``, the log Probability
            Density Function (log PDF) of the proposal given the current value.
        """
        raise NotImplementedError

    def get_proposal_function(self, func_name='getProposal'):
        """Get a proposal function that returns proposals for a requested parameter.

        Args:
            func_name (str): the CL function name of the returned function

        Returns:
            str: A function with the signature:
                .. code-block:: c

                    mot_float_type <func_name>(const int i, const mot_float_type current, ranluxcl_state_t* ranluxclstate,
                                               mot_float_type* const proposal_state)

            Where ``i`` is the index of the parameter for which we want the proposal and ``current`` is the current
            value of that parameter. The argument ``proposal_state`` is the state of the proposal distribution.
            One can obtain random numbers with:
            .. code-block:: c

                float4 randomnr = ranluxcl(ranluxclstate);
        """
        raise NotImplementedError

    def get_proposal_state_update_function(self, func_name='updateProposalState'):
        """Get a function that can update the parameters of the proposals

        Args:
            func_name (str): the CL function name of the returned function

        Returns:
            str: A function with the signature:
                .. code-block:: c

                    void <func_name>(uint* const ac_between_proposal_updates, const uint proposal_update_intervals,
                                     mot_float_type* const proposal_state);

            Where ``ac_between_proposal_updates`` is the acceptance count in between proposal updates,
            ``proposal_update_intervals`` is the interval at which we update the proposals and ``proposal_state``
            is the current list of proposal parameters (to be updated in place). Please note that
            ``ac_between_proposal_updates`` is per sampled parameter while ``proposal_state`` is
            per proposal parameter. If the number of parameters of a proposal function is not 1, than the
            two arrays do not share the same indices.
        """
        raise NotImplementedError

    def get_log_prior_function(self, func_name='getLogPrior'):
        """Get the prior function that returns a double representing the prior information about the given parameters.

        The prior function must be in log space.

        Args:
            func_name (str): the CL function name of the returned function

        Returns:
            str: A function with the signature:
                .. code-block:: c

                    mot_float_type <func_name>(const mot_float_type* const x);

            Which is called by the sampling routine to calculate the posterior probability.
        """
        raise NotImplementedError

    def samples_to_statistics(self, samples_dict):
        """Create statistics out of the given set of samples (in a dictionary).

        Args:
            samples_dict (dict): Keys being the parameter names, values the roi list in 2d
                (1st dim. is voxel, 2nd dim. is samples).

        Returns:
            dict: The same dictionary but with statistical maps (mean, avg etc.) for each parameter, instead of the raw
            samples. In essence this is where one can place the logic to go from samples to meaningful maps.
        """
        raise NotImplementedError
