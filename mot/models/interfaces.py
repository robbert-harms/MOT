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

    def get_problems_var_data(self):
        """Get a dict with all the data per problem.

        As an example, suppose per problem instance we have the data named 'observations'.
        We should then return something like: {'observations': ndarray((<nmr_problems>, <number of items>))}

        Returns:
            dict: A dictionary where each element holds a 2d numpy array.
                That array should contains for each problem (first dimension / rows) a
                number of items (second dimension / columns) that is used in the evaluation function.
        """

    def get_problems_prtcl_data(self):
        """Get a dict with the data that is constant for each problem (in Diffusion MRI the protocol/scheme), but
        differs per measurement.

        Returns:
            dict: A dict where each element holds an numpy array that is used in the evaluation function
                and each key is used as name in the cl data struct
        """

    def get_problems_fixed_data(self):
        """Get a dict with all the fixed data. These are fixed for all voxels, for all measurement points.

        Returns:
            dict: The dict with all the fixed data. The names of the keys are used in the CL cl data structs.
        """

    def get_nmr_problems(self):
        """Get the number of problems (number of voxels in MRI).

        Returns:
            int: A single integer specifying the number of problem instances
        """

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
                double <func_name>(const optimize_data* const data, const double* const x, const int observation_index);
        """

    def get_observation_return_function(self, func_name='getObservation'):
        """Get the CL function that returns the observation for the given problem.

        Args:
            func_name (string): specifies the name of the function.

        Returns:
            str: An CL function with the signature:
                double <func_name>(const optimize_data* const data, const int observation_index);
        """

    def get_objective_function(self, func_name="calculateObjective"):
        """Get the objective function that evaluates the entire problem instance under a noise model.

        Args:
            func_name (string): specifies the name of the function.

        Returns:
            A function of the kind:
                double <func_name>(const optimize_data* const data, double* const x);
        """

    def get_initial_parameters(self, results_dict=None):
        """Get a two dimensional matrix with the initial parameters (starting points) for every voxel.

        Optionally, one may specify a list of previously calculated results which may be applicable to the model.
        If a parameter is found in the results_dict, those values are used for the initial parameters.

        Args:
            results_dict (dict): a dictionary with for every parameter name, a value per voxel which is (for example)
                the result of a previous calculation.

        Returns:
            array like: A two dimensional matrix with on the first axis the the problem instances and on the second
                the parameter values per problem instance
        """

    def get_lower_bounds(self):
        """Get for each estimable parameter the lower bounds.

        Returns:
            An numpy row with on each column a single value, the lower bound for that parameter. This value
            can be the literal string '-inf' for infinity.
        """

    def get_upper_bounds(self):
        """Get for each estimable parameter the upper bounds.

        Returns:
            An numpy row with on each column a single value, the upper bound for that parameter. This value
            can be the literal string 'inf' for infinity.
        """

    def get_optimized_param_names(self):
        """Get a list of names with the free parameter names (the parameters that are estimated by the routines).

        The function get_optimization_output_param_names() returns the names of all the parameter names,
        including fixed and static parameters. This should only return the names of the parameters
        that are actually used in the optimization.

        Returns:
            A list with the parameter names (in dot format) of all the estimated (free) parameters.
        """

    def get_optimization_output_param_names(self):
        """Get a list with the names of the parameters, this is the list of keys to the titles and results.

        See get_optimized_param_names() for getting the names of the parameters that are actually being optimized.

        This should be a complete overview of all the maps returned from optimizing this model.

        Returns:
            list of str: a list with the parameter names
        """

    def get_nmr_inst_per_problem(self):
        """Get the number of instances/data points per problem. In DMRI this is the length of the protocol.

        The minimum is one instance per problem.

        This number represents the number of data points

        Returns:
            int: A single integer specifying the number of instances per problem.
        """

    def get_nmr_estimable_parameters(self):
        """Get the number of estimable parameters.

        Returns:
            int: A single integer specifying the number of estimable parameters
        """

    def get_parameter_codec(self):
        """Get the parameter codec from this model. This should be an implementation of AbstractCodec.

        Returns:
            An abstract codec model that holds the CL code for the codec transformations. This function may also
            return None, which indicates that no parameter codec is supposed to be used.
        """

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
                void <func_name>(const optimize_data* data, double* x);

            Which is called for every voxel and must in place edit the x variable.
        """

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

    def get_log_likelihood_function(self, func_name="getLogLikelihood"):
        """Get the Log Likelihood function that evaluates the entire problem instance under a noise model

        Args:
            func_name (string): specifies the name of the function.

        Returns:
            str: A function of the kind:
                double <func_name>(const optimize_data* const data, double* const x);
        """

    def is_proposal_symmetric(self):
        """Check if the entire proposal distribution is symmetric ( q(x|x') == q(x'|x) ).

        Returns:
            boolean: True if the proposal distribution is symmetric, false otherwise.
        """

    def get_proposal_logpdf(self, func_name='getProposalLogPDF'):
        """Get the probability density function of the proposal in log space.

        Returns:
            A function with the signature:
                double <func_name>(const int i, const double proposal, const double current, double* const parameters)

            Where i is the index of the parameter we would like to get the proposal from, current is the current
            value of that parameter and proposal the proposal value of the parameter. It should return for the requested
            parameter a value q(proposal | current). That is, the probability density function of the proposal given
            the current value (in log space).
            Parameters is the list of adaptable parameters.
        """

    def get_proposal_function(self, func_name='getProposal'):
        """Get a proposal function that returns proposals for a requested parameter.

        Returns:
            A function with the signature:
                double <func_name>(const int i, const double current, ranluxcl_state_t* ranluxclstate,
                                   double* const parameters)

            Where i is the index of the parameter we would like to get the proposal from and current is the current
            value of that parameter. One can obtain random numbers with:
                float4 randomnr = ranluxcl(ranluxclstate);
            Parameters is the list of adaptable parameters.
        """

    def get_proposal_parameters_update_function(self, func_name='updateProposalParameters'):
        """Get a function that can update the parameters of the proposals

        Returns:
            A function with the signature:
                void <func_name>(uint* const ac_between_proposal_updates, const uint proposal_update_intervals,
                                 double* const proposal_parameters);

            Where ac_between_proposal_updates is the acceptance count in between proposal updates,
            proposal_update_intervals is the interval at which we update the proposals and proposal_parameters
            is the current list of proposal parameters (to be updated in place),

            Please note that the ac_between_proposal_updates is per sampled parameter while the proposal_parameters is
            per proposal parameter. If the number of parameters of a proposal function is not equal to one then those
            two arrays do not share the same index.
        """

    def get_log_prior_function(self, func_name='getLogPrior'):
        """Get the prior function that returns a double representing the prior information about the given parameters.

        The prior function must be in log space.

        Returns:
            A function of the kind:
                double <func_name>(const double* const x);

            Which is called by the sampling routine to calculate the posterior probability.
        """

    def samples_to_statistics(self, samples_dict):
        """Create statistics out of the given set of samples (in a dictionary).

        Args:
            samples_dict (dict):
                Keys being the parameter names, values the roi list in 2d (1st dim. is voxel, 2nd dim. is samples).

        Returns:
            The same dictionary but with statistical maps (mean, avg etc.) for each parameter, instead of the raw
            samples. In essence this is where one can place the logic to go from samples to meaningful maps.
        """


class SmoothableModelInterface(OptimizeModelInterface):

    def smooth(self, results, filter_routine):
        """Smooth the given results according to the rules in this model and the given smoother.

        This is meant for spatial smoothing across the problems.

        There is a reason we have this kind of interface, instead of a function 'get_smoothable_mapnames' and then
        do the smoothing in the calling class. That is because the optimization classes work with a list of problems
        and do not know the spatial locations of each problem in the original dataset.

        Args:
            results (dict): Parameters and 1d maps with the voxel maps.
            filter_routine (AbstractFilter): A smoothing routine object.

        Returns:
            results (dict): A dictionary with the same keys as the given results,
                but with updated values. It may contain references to elements in the given results dictionary.
        """


class PerturbationModelInterface(OptimizeModelInterface):

    def perturbate(self, results):
        """Perturbate the given results according to the rules in this model.

        Args:
            results (dict): Parameters and 1d maps with the voxel maps.

        Returns:
            results (dict): A dictionary with the same keys as the given results,
                but with perturbated values. This can be performed in-place.
        """