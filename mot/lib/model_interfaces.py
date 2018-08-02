"""The model interfaces.

This encapsulates all the information we need about models to be able to optimize or sample them using the routines
in MOT. These interfaces expose data and modeling code. The data is represented as :class:`mot.lib.utils.KernelInputData`
instances and the CL code as strings.
"""
from mot.lib.cl_function import SimpleCLFunction

__author__ = 'Robbert Harms'
__date__ = "2014-03-14"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ModelBasicInfoInterface(object):

    def get_kernel_data(self):
        """Return a dictionary of input data objects we need to load into the kernel.

        Returns:
            dict[str: mot.lib.utils.KernelData]: the list of data objects we need to load in the kernel
        """
        raise NotImplementedError()

    def get_nmr_observations(self):
        """Get the number of instances/data points per problem.

        The minimum is one instance per problem.

        This number represents the number of data points

        Returns:
            int: the number of instances per problem.
        """
        raise NotImplementedError()


class OptimizeModelInterface(ModelBasicInfoInterface):

    def get_objective_function(self):
        """Get the objective function that returns the objective value and optionally, the objectives per observation.

        In the case that any of the objective lists are non-null pointers, they should be filled with the
        objective function values per observation.

        Returns:
            mot.lib.cl_function.CLFunction: A CL function with signature:

                .. code-block:: c

                    double <func_name>(mot_data_struct* data,
                                       local const mot_float_type* const x,
                                       local mot_float_type* objective_list,
                                       local double* objective_value_tmp);

                The objective list needs to be filled when the pointer is not null.
                This method should use the given local memory for the summation reduction.
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


class SampleModelInterface(ModelBasicInfoInterface):

    def get_log_likelihood_function(self):
        """Get the (complete) CL Log Likelihood function that evaluates the given instance under a noise model.

        Returns:
            mot.lib.cl_function.CLFunction: A function of the kind:

                .. code-block:: c

                    double <func_name>(mot_data_struct* data,
                                       local const mot_float_type* const x,
                                       local double* objective_value_tmp);
        """
        raise NotImplementedError()

    def get_log_prior_function(self):
        """Get the prior function that returns the prior information about the given parameters.

        The prior function must be in log space.

        Returns:
            mot.lib.cl_function.CLFunction: A function with the signature:
                .. code-block:: c

                    mot_float_type <func_name>(
                        mot_data_struct* data,
                        local const mot_float_type* const x
                    );

            Which is called by the sample routine to calculate the prior probability.
        """
        raise NotImplementedError()

    def get_finalize_proposal_function(self):
        """Get a CL function that is called for every new proposal, in order to finalize it.

        This allows the model to change a proposal before computing the prior or likelihood probabilities.

        As an example, suppose you are sample a polar coordinate :math:`\theta` defined on
        :math:`[0, 2\pi]` with a random walk Metropolis proposal distribution. This distribution might propose positions
        outside of the range of :math:`\theta`. Of course the model function could deal with that by taking the modulus
        of the input, but then you have to post-process the chain with the same transformation. Instead, this function
        allows changing the proposal before it is put into the model and before it is stored.

        Returns:
            mot.lib.cl_function.CLFunction: A function with the signature:
                .. code-block:: c

                    void <func_name>(
                        mot_data_struct* data,
                        local mot_float_type* x
                    );

            Which is called by the sample routine to finalize the proposal.
        """
        return SimpleCLFunction(
            'void', 'finalizeProposal',
            ['mot_data_struct* data', 'local const mot_float_type* const x'],
            '')


class NumericalDerivativeInterface(OptimizeModelInterface):
    """Extends an optimization model for calculating numerical derivatives of the objective function.

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
            mot.lib.cl_function.CLFunction: A function with the signature:
                .. code-block:: c

                    void <func_name>(mot_data_struct* data, local mot_float_type* params);

                Where the data is the kernel data struct and params is the vector with the suggested parameters and
                which can be modified in place. Note that this is called two times, one with the parameters plus
                the step and one time without.
        """
        raise NotImplementedError()
