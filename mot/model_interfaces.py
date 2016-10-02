__author__ = 'Robbert Harms'
__date__ = "2014-03-14"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class OptimizeModelInterface(object):

    @property
    def name(self):
        """Get the name of this model. This should be overwrited by the implementing model.

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
            {'observations': SimpleDataAdapter(ndarray((<nmr_problems>, <number of items>)), ...)}

        Returns:
            dict: A dictionary where each element holds a SimpleDataAdapter from which to get the data
                The data in the adapter should contain for each problem (first dimension / rows) a
                number of items (second dimension / columns) that is used in the evaluation function.
        """
        raise NotImplementedError

    def get_problems_protocol_data(self):
        """Get a dict with the data that is constant for each problem (in Diffusion MRI the protocol/scheme), but
        differs per measurement.

        Returns:
            dict: A dict where each element holds a SimpleDataAdapter that is used in the evaluation function.
                Each key is used as name in the cl data struct.
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
                mot_float_type <func_name>(const optimize_data* const data, const int observation_index);
        """
        raise NotImplementedError

    def get_objective_function(self, func_name="calculateObjective"):
        """Get the objective function that evaluates the entire problem instance under a noise model.

        Args:
            func_name (string): specifies the name of the function.

        Returns:
            A CL function with signature:
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
            A CL function with signature:
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
            array like: A two dimensional matrix with on the first axis the problem instances and on the second
                the parameter values per problem instance
        """
        raise NotImplementedError

    def get_lower_bounds(self):
        """Get for each estimable parameter the lower bounds.

        Returns:
            An numpy row with on each column a single value, the lower bound for that parameter. This value
            can be the literal string '-inf' for infinity.
        """
        raise NotImplementedError

    def get_upper_bounds(self):
        """Get for each estimable parameter the upper bounds.

        Returns:
            An numpy row with on each column a single value, the upper bound for that parameter. This value
            can be the literal string 'inf' for infinity.
        """
        raise NotImplementedError

    def get_optimized_param_names(self):
        """Get a list of names with the free parameter names (the parameters that are estimated by the routines).

        The function get_optimization_output_param_names() returns the names of all the parameter names,
        including fixed and static parameters. This should only return the names of the parameters
        that are actually used in the optimization.

        Returns:
            A list with the parameter names (in dot format) of all the estimated (free) parameters.
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
            int: A single integer specifying the number of instances per problem.
        """
        raise NotImplementedError

    def get_nmr_estimable_parameters(self):
        """Get the number of estimable parameters.

        Returns:
            int: A single integer specifying the number of estimable parameters
        """
        raise NotImplementedError

    def get_parameter_codec(self):
        """Get the parameter codec from this model. This should be an implementation of AbstractCodec.

        Returns:
            An abstract codec model that holds the CL code for the codec transformations. This function may also
            return None, which indicates that no parameter codec is supposed to be used.
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
            Return None if this function is not used, else a function of the kind:
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
        super(SampleModelInterface, self).__init__()

    def get_proposal_parameter_values(self):
        """Get a list of parameter value for the adaptable proposal parameters.

        Returns:
            list: list of double values with the proposal parameter values that are adaptable.
        """
        raise NotImplementedError

    def get_log_likelihood_function(self, func_name="getLogLikelihood", evaluation_model=None, full_likelihood=True):
        """Get the Log Likelihood function that evaluates the entire problem instance under a noise model

        Args:
            func_name (string): specifies the name of the function.
            evaluation_model (EvaluationModel): the evaluation model to use for the log likelihood. If not given
                we use the one defined in the model.
            full_likelihood (boolean): if we want the complete likelihood, or if we can drop the constant terms.
                The default is the complete likelihood. Disable for speed.

        Returns:
            str: A function of the kind:
                mot_float_type <func_name>(const optimize_data* const data, mot_float_type* const x);
        """
        raise NotImplementedError

    def is_proposal_symmetric(self):
        """Check if the entire proposal distribution is symmetric ( q(x|x') == q(x'|x) ).

        Returns:
            boolean: True if the proposal distribution is symmetric, false otherwise.
        """
        raise NotImplementedError

    def get_proposal_logpdf(self, func_name='getProposalLogPDF'):
        """Get the probability density function of the proposal in log space.

        Args:
            func_name (str): the CL function name of the returned function

        Returns:
            A function with the signature:
                mot_float_type <func_name>(const int i, const mot_float_type proposal,
                                        const mot_float_type current, mot_float_type* const parameters)

            Where i is the index of the parameter we would like to get the proposal from, current is the current
            value of that parameter and proposal the proposal value of the parameter. It should return for the requested
            parameter a value q(proposal | current). That is, the probability density function of the proposal given
            the current value (in log space).
            Parameters is the list of adaptable parameters.
        """
        raise NotImplementedError

    def get_proposal_function(self, func_name='getProposal'):
        """Get a proposal function that returns proposals for a requested parameter.

        Args:
            func_name (str): the CL function name of the returned function

        Returns:
            A function with the signature:
                mot_float_type <func_name>(const int i, const mot_float_type current, ranluxcl_state_t* ranluxclstate,
                                   mot_float_type* const parameters)

            Where i is the index of the parameter we would like to get the proposal from and current is the current
            value of that parameter. One can obtain random numbers with:
                float4 randomnr = ranluxcl(ranluxclstate);
            Parameters is the list of adaptable parameters.
        """
        raise NotImplementedError

    def get_proposal_parameters_update_function(self, func_name='updateProposalParameters'):
        """Get a function that can update the parameters of the proposals

        Args:
            func_name (str): the CL function name of the returned function

        Returns:
            A function with the signature:
                void <func_name>(uint* const ac_between_proposal_updates, const uint proposal_update_intervals,
                                 mot_float_type* const proposal_parameters);

            Where ac_between_proposal_updates is the acceptance count in between proposal updates,
            proposal_update_intervals is the interval at which we update the proposals and proposal_parameters
            is the current list of proposal parameters (to be updated in place),

            Please note that the ac_between_proposal_updates is per sampled parameter while the proposal_parameters is
            per proposal parameter. If the number of parameters of a proposal function is not equal to one then those
            two arrays do not share the same index.
        """
        raise NotImplementedError

    def get_log_prior_function(self, func_name='getLogPrior'):
        """Get the prior function that returns a double representing the prior information about the given parameters.

        The prior function must be in log space.

        Args:
            func_name (str): the CL function name of the returned function

        Returns:
            A function of the kind:
                mot_float_type <func_name>(const mot_float_type* const x);

            Which is called by the sampling routine to calculate the posterior probability.
        """
        raise NotImplementedError

    def samples_to_statistics(self, samples_dict):
        """Create statistics out of the given set of samples (in a dictionary).

        Args:
            samples_dict (dict):
                Keys being the parameter names, values the roi list in 2d (1st dim. is voxel, 2nd dim. is samples).

        Returns:
            The same dictionary but with statistical maps (mean, avg etc.) for each parameter, instead of the raw
            samples. In essence this is where one can place the logic to go from samples to meaningful maps.
        """
        raise NotImplementedError
