import numpy as np
from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import KernelInputArray, SimpleNamedCLFunction, KernelInputLocalMemory, KernelInputAllocatedOutput, \
    multiprocess_mapping
from ...cl_routines.base import CLRoutine
from scipy import linalg
import warnings


__author__ = 'Robbert Harms'
__date__ = '2017-10-16'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class NumericalHessian(CLRoutine):

    def calculate(self, model, parameters, double_precision=False, step_ratio=2, nmr_steps=15, step_offset=None):
        """Calculate and return the Hessian of the given function at the given parameters.

        This calculates the Hessian using central difference (using a 2nd order Taylor expansion) with a Richardson
        extrapolation over the proposed sequence of steps.

        The Hessian is evaluated at the steps:

        .. math::
            \quad  ((f(x + d_j e_j + d_k e_k) - f(x + d_j e_j - d_k e_k)) -
                    (f(x - d_j e_j + d_k e_k) - f(x - d_j e_j - d_k e_k)) /
                    (4 d_j d_k)

        where :math:`e_j` is a vector where element :math:`j` is one and the rest are zero
        and :math:`d_j` is a scalar spacing :math:`steps_j`.

        Steps are generated according to a exponentially diminishing ratio defined as:

            steps = max_step * step_ratio**-(i+offset), i=0, 1,.., nmr_steps-1.

        Where the max step is taken from the model. For example, a maximum step of 2 with a step ratio of 2 and with
        4 steps gives: [2.0, 1.0, 0.5, 0.25]. If offset would be 2, we would instead get: [0.5, 0.25, 0.125, 0.0625].

        If number of steps is 1, we use not Richardson extrapolation and return the results of the first step. If the
        number of steps is 2 we use a first order Richardson extrapolation step. For all higher number of steps
        we use a second order Richardson extrapolation.

        Args:
            model (mot.model_interfaces.NumericalDerivativeInterface): the model to use for calculating the
                derivative.
            parameters (ndarray): The parameters at which to evaluate the gradient. A (d, p) matrix with d problems,
                p parameters and n samples.
            double_precision (boolean): if we want to calculate everything in double precision
            step_ratio (float): the ratio at which the steps diminish.
            nmr_steps (int): the number of steps we will generate. We will calculate the derivative for each of these
                step sizes and extrapolate the best step size from among them. The minimum number of steps is 2.
            step_offset (the offset in the steps, if set we start the steps from the given offset.

        Returns:
            ndarray: the gradients for each of the parameters for each of the problems
        """
        if len(parameters.shape) == 1:
            parameters = parameters[None, :]
        nmr_params = parameters.shape[1]

        parameter_scalings = np.array(model.numdiff_get_scaling_factors())

        derivatives = self._compute_derivatives(model, parameters, double_precision, step_ratio, step_offset, nmr_steps)

        if nmr_steps == 1:
            return self._results_vector_to_matrix(derivatives[..., 0], nmr_params) \
                   * np.outer(parameter_scalings, parameter_scalings)

        richardson_derivatives, richardson_errors = self._richardson_convolutions(
            derivatives, step_ratio, double_precision)

        if nmr_steps == 2:
            return self._results_vector_to_matrix(richardson_derivatives[..., 0], nmr_params) \
                   * np.outer(parameter_scalings, parameter_scalings)

        extrapolate = Extrapolate(step_ratio)

        def data_iterator():
            for problem_ind in range(parameters.shape[0]):
                yield (richardson_derivatives[problem_ind].T,
                       richardson_errors[problem_ind].T,
                       nmr_params)

        # result = np.array(list(map(extrapolate, data_iterator())))
        result = np.array(multiprocess_mapping(extrapolate, data_iterator()))
        return self._results_vector_to_matrix(result, nmr_params) * np.outer(parameter_scalings, parameter_scalings)

    def _compute_derivatives(self, model, parameters, double_precision, step_ratio, step_offset, nmr_steps):
        """Compute the second derivative using the central difference method.

        This calculates for the given step the numerical derivative and returns that value directly.
        """
        parameter_scalings = np.array(model.numdiff_get_scaling_factors())

        nmr_params = parameters.shape[1]
        nmr_derivatives = (nmr_params ** 2 - nmr_params) // 2 + nmr_params

        initial_step = self._get_initial_step_size(model, parameters)
        if step_offset:
            initial_step *= float(step_ratio) ** -step_offset

        all_kernel_data = dict(model.get_kernel_data())
        all_kernel_data.update({
            'parameters': KernelInputArray(parameters, ctype='mot_float_type'),
            'local_reduction_lls': KernelInputLocalMemory('double'),
            'parameter_scalings_inv': KernelInputArray(1. / parameter_scalings, ctype='double', offset_str='0'),
            'initial_step': KernelInputArray(initial_step, ctype='mot_float_type', is_writable=True),
            'step_evaluates': KernelInputAllocatedOutput((parameters.shape[0], nmr_derivatives, nmr_steps), 'double'),
        })

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._get_single_step_kernel(model, nmr_params, nmr_steps, step_ratio),
                             all_kernel_data, parameters.shape[0], double_precision=double_precision,
                             use_local_reduction=True)

        return all_kernel_data['step_evaluates'].get_data()

    def _richardson_convolutions(self, derivatives, step_ratio, double_precision):
        nmr_problems, nmr_derivatives, nmr_steps = derivatives.shape
        richardson_coefficients = self._get_richardson_coefficients(step_ratio, min(nmr_steps, 3) - 1)
        nmr_convolutions = nmr_steps - (len(richardson_coefficients) - 2)
        final_nmr_convolutions = nmr_convolutions - 1

        kernel_data = {
            'derivatives': KernelInputArray(derivatives, 'double', offset_str='{problem_id} * ' + str(nmr_steps)),
            'richardson_convolutions': KernelInputAllocatedOutput(
                (nmr_problems * nmr_derivatives, nmr_convolutions), 'double'),
            'errors': KernelInputAllocatedOutput(
                (nmr_problems * nmr_derivatives, final_nmr_convolutions), 'double'),
        }

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._richardson_error_kernel(nmr_steps, nmr_convolutions, richardson_coefficients),
                             kernel_data, nmr_problems * nmr_derivatives, double_precision=double_precision,
                             use_local_reduction=False)

        convolutions = np.reshape(kernel_data['richardson_convolutions'].get_data(),
                          (nmr_problems, nmr_derivatives, nmr_convolutions))
        errors = np.reshape(kernel_data['errors'].get_data(),
                            (nmr_problems, nmr_derivatives, final_nmr_convolutions))

        return convolutions[..., :final_nmr_convolutions], errors

    def _results_vector_to_matrix(self, vectors, nmr_params):
        """Transform for every problem (and optionally every step size) the vector results to a square matrix.

        Since we only process the lower triangular items, we have to convert the 1d vectors into 2d matrices for every
        problem. This function does that.

        Args:
            vectors (ndarray): for every problem (and step size) the 1d vector with results
            matrices (ndarray): for every problem the 2d square matrix with the results placed in matrix form
        """
        matrices = np.zeros(vectors.shape[:-1] + (nmr_params, nmr_params), dtype=vectors.dtype)

        ltr_ind = 0
        for px in range(nmr_params):
            matrices[..., px, px] = vectors[..., ltr_ind]
            ltr_ind += 1

            for py in range(px + 1, nmr_params):
                matrices[..., px, py] = vectors[..., ltr_ind]
                matrices[..., py, px] = vectors[..., ltr_ind]
                ltr_ind += 1
        return matrices

    def _get_initial_step_size(self, model, parameters):
        """Get an initial step size to use for every parameter.

        This chooses the maximum of the maximum step defined in the model and the minimum step possible
        for a parameter given its bounds.

        Args:
            model (mot.model_interfaces.NumericalDerivativeInterface): the model to use for calculating the
                derivative.
            parameters (ndarray): The parameters at which to evaluate the gradient. A (d, p) matrix with d problems,
                p parameters and n samples.

        Returns:
            ndarray: for every problem instance the vector with the initial step size for each parameter.
        """
        upper_bounds = model.get_upper_bounds()
        lower_bounds = model.get_lower_bounds()

        max_step = model.numdiff_get_max_step()
        minimum_allowed_step = np.minimum(np.abs(parameters - lower_bounds),
                                          np.abs(np.array(upper_bounds) - parameters)) \
                               * model.numdiff_get_scaling_factors()

        initial_step = np.zeros_like(minimum_allowed_step)

        for ind in range(parameters.shape[1]):
            if model.numdiff_use_bounds()[ind]:
                initial_step[:, ind] = np.minimum(minimum_allowed_step[:, ind], max_step[ind])
            else:
                initial_step[:, ind] = max_step[ind]

        return initial_step

    def _get_richardson_coefficients(self, step_ratio, nmr_extrapolations):
        """Get the matrix with the convolution rules.

        In the kernel we will extrapolate the sequence based on the Richardsons method.

        This method assumes we have a series expansion like:

            L = f(h) + a0 * h^p_0 + a1 * h^p_1+ a2 * h^p_2 + ...

        where p_i = order + step * i  and f(h) -> L as h -> 0, but f(0) != L.

        If we evaluate the right hand side for different stepsizes h we can fit a polynomial to that sequence
        of approximations.

        Instead of using all the generated points at the same time, we convolute the Richardson method over the
        acquired steps to approximate the higher order error terms.

        Args:
            step_ratio (float): the ratio at which the steps diminish.
            nmr_extrapolations (int): the number of extrapolations we want to do. Each extrapolation requires
                an evaluation at an exponentially decreasing step size.

        Returns:
            ndarray: a vector with the extrapolation coefficients.
        """
        if nmr_extrapolations == 0:
            return np.array([1])

        error_diminishing_per_step = 2
        taylor_expansion_order = 2

        def r_matrix(num_terms):
            i, j = np.ogrid[0:num_terms + 1, 0:num_terms]
            r_mat = np.ones((num_terms + 1, num_terms + 1))
            r_mat[:, 1:] = (1.0 / step_ratio) ** (i * (error_diminishing_per_step * j + taylor_expansion_order))
            return r_mat

        return linalg.pinv(r_matrix(nmr_extrapolations))[0]

    def _get_compute_functions_cl(self, model, nmr_params, nmr_steps, step_ratio):
        ll_function = model.get_objective_per_observation_function()
        numdiff_param_transform = model.numdiff_parameter_transformation()

        nmr_inst_per_problem = model.get_nmr_inst_per_problem()

        func = ll_function.get_cl_code()
        func += numdiff_param_transform.get_cl_code()

        return func + '''
            double _calculate_function(mot_data_struct* data, mot_float_type* x){
                ulong observation_ind;
                ulong local_id = get_local_id(0);
                data->local_reduction_lls[local_id] = 0;
                uint workgroup_size = get_local_size(0);
                uint elements_for_workitem = ceil(''' + str(nmr_inst_per_problem) + '''
                                                  / (mot_float_type)workgroup_size);

                if(workgroup_size * (elements_for_workitem - 1) + local_id >= ''' + str(nmr_inst_per_problem) + '''){
                    elements_for_workitem -= 1;
                }

                for(uint i = 0; i < elements_for_workitem; i++){
                    observation_ind = i * workgroup_size + local_id;
                    data->local_reduction_lls[local_id] += ''' + ll_function.get_cl_function_name() + '''(
                        data, x, observation_ind);
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                
                double ll = 0;
                for(uint i = 0; i < workgroup_size; i++){
                    ll += data->local_reduction_lls[i];
                }
                return ll;
            }
            
            /**
             * Evaluate the model with a perturbation in two dimensions.
             *
             * Args:
             *  data: the data container
             *  x_input: the array with the input parameters
             *  perturb_dim_0: the index (into the x_input parameters) of the first parameter to perturbate
             *  perturb_0: the added perturbation of the index corresponding to ``perturb_dim_0``
             *  perturb_dim_1: the index (into the x_input parameters) of the second parameter to perturbate
             *  perturb_1: the added perturbation of the index corresponding to ``perturb_dim_1``
             *
             * Returns:
             *  the function evaluated at the parameters plus their perturbation.
             */
            double _eval_step(mot_data_struct* data, mot_float_type* x_input,
                              uint perturb_dim_0, mot_float_type perturb_0,
                              uint perturb_dim_1, mot_float_type perturb_1){

                mot_float_type x_tmp[''' + str(nmr_params) + '''];
                for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                    x_tmp[i] = x_input[i];
                }
                x_tmp[perturb_dim_0] += perturb_0;
                x_tmp[perturb_dim_1] += perturb_1;

                ''' + numdiff_param_transform.get_cl_function_name() + '''(data, x_tmp);
                return _calculate_function(data, x_tmp);
            }
            
            /**
             * Compute the 2nd derivative of one element of the Hessian for a number of steps.
             */
            void _compute_steps(mot_data_struct* data, mot_float_type* x_input, mot_float_type f_x_input,
                                uint px, uint py, global double* step_evaluate_ptr){
                
                double step_x;
                double step_y;
                double tmp;
                bool is_first_workitem = get_local_id(0) == 0;
                
                if(px == py){
                    for(uint step_ind = 0; step_ind < ''' + str(nmr_steps) + '''; step_ind++){
                        step_x = data->initial_step[px] / pown(''' + str(float(step_ratio)) + ''', step_ind);
                        
                        tmp = (
                              _eval_step(data, x_input,
                                         px, 2 * (step_x * data->parameter_scalings_inv[px]),
                                         0, 0)
                            + _eval_step(data, x_input,
                                         px, -2 * (step_x * data->parameter_scalings_inv[px]),
                                         0, 0)
                            - 2 * f_x_input
                        ) / (4 * step_x * step_x);
                        
                        if(is_first_workitem){
                            step_evaluate_ptr[step_ind] = tmp;
                        }
                    }
                }
                else{
                    for(uint step_ind = 0; step_ind < ''' + str(nmr_steps) + '''; step_ind++){
                        step_x = data->initial_step[px] / pown(''' + str(float(step_ratio)) + ''', step_ind);
                        step_y = data->initial_step[py] / pown(''' + str(float(step_ratio)) + ''', step_ind);
                        
                        tmp = (
                              _eval_step(data, x_input,
                                         px, step_x * data->parameter_scalings_inv[px],
                                         py, step_y * data->parameter_scalings_inv[py])
                            - _eval_step(data, x_input,
                                         px, step_x * data->parameter_scalings_inv[px],
                                         py, -step_y * data->parameter_scalings_inv[py])
                            - _eval_step(data, x_input,
                                         px, -step_x * data->parameter_scalings_inv[px],
                                         py, step_y * data->parameter_scalings_inv[py])
                            + _eval_step(data, x_input,
                                         px, -step_x * data->parameter_scalings_inv[px],
                                         py, -step_y * data->parameter_scalings_inv[py])
                        ) / (4 * step_x * step_y);
    
                        if(is_first_workitem){
                            step_evaluate_ptr[step_ind] = tmp;
                        }                       
                    }
                }
            }
        '''

    def _get_error_estimate_functions_cl(self, nmr_steps, nmr_convolutions,
                                         richardson_coefficients):
        func = '''
            /**
             * Apply a simple kernel convolution over the results from each row of steps.
             *
             * This applies a convolution starting from the starting step index that contained a valid move.
             * It uses the mode 'reflect' to deal with outside points.
             *
             * Please note that this kernel is hard coded to work with 2nd order Taylor expansions derivatives only.

             * Args:
             *  step_evaluates: the step evaluates for each row of the step sizes
             *  richardson_extrapolations: the array to place the convoluted results in
             */
            void _apply_richardson_convolution(
                    global double* derivatives,
                    global double* richardson_convolutions){

                double convolution_kernel[''' + str(len(richardson_coefficients)) + '''] = {''' + \
                    ', '.join(map(str, richardson_coefficients)) + '''};
                
                for(uint step_ind = 0; step_ind < ''' + str(nmr_convolutions) + '''; step_ind++){
                    
                    // convolute the Richardson coefficients
                    for(uint kernel_ind = 0; kernel_ind < ''' + str(len(richardson_coefficients)) + '''; kernel_ind++){
                        
                        uint kernel_step_ind = step_ind + kernel_ind;

                        // reflect
                        if(kernel_step_ind >= ''' + str(nmr_steps) + '''){
                            kernel_step_ind -= 2 * (kernel_step_ind - ''' + str(nmr_steps) + ''') + 1;
                        }

                        richardson_convolutions[step_ind] +=
                            derivatives[kernel_step_ind] * convolution_kernel[kernel_ind];
                    }
                }
            }
            
            /**
             * Compute the errors from using the Richardson extrapolation.
             *
             * "A neat trick to compute the statistical uncertainty in the estimate of our desired derivative is to use 
             *  statistical methodology for that error estimate. While I do appreciate that there is nothing truly 
             *  statistical or stochastic in this estimate, the approach still works nicely, providing a very 
             *  reasonable estimate in practice. A three term Richardson-like extrapolant, then evaluated at four 
             *  distinct values for \delta, will yield an estimate of the standard error of the constant term, with one 
             *  spare degree of freedom. The uncertainty is then derived by multiplying that standard error by the 
             *  appropriate percentile from the Students-t distribution." 
             * 
             * Cited from https://numdifftools.readthedocs.io/en/latest/src/numerical/derivest.html
             * 
             * In addition to the error derived from the various estimates, we also approximate the numerical round-off 
             * errors in this method. All in all, the resulting errors should reflect the absolute error of the
             * estimates.
             */ 
            void _compute_richardson_errors(
                    global double* derivatives, 
                    global double* richardson_convolutions,
                    global double* errors){
                
                //the magic number 12.7062... follows from the student T distribution with one dof. 
                // Example computation: 
                // >>> import scipy.stats as ss
                // >>> allclose(ss.t.cdf(12.7062047361747, 1), 0.975) # True
                double fact = max(
                    (mot_float_type)''' + str(12.7062047361747 * np.sqrt(np.sum(richardson_coefficients**2))) + ''',
                    (mot_float_type)MOT_EPSILON * 10);
                
                double tolerance;
                double error;
                
                for(uint conv_ind = 0; conv_ind < ''' + str(nmr_convolutions - 1) + '''; conv_ind++){
                    tolerance = max(fabs(richardson_convolutions[conv_ind + 1]), 
                                    fabs(richardson_convolutions[conv_ind])
                                    ) * MOT_EPSILON * fact;
                    
                    error = fabs(richardson_convolutions[conv_ind] - richardson_convolutions[conv_ind + 1]) * fact;
                    
                    if(error <= tolerance){
                        error += tolerance * 10;
                    }
                    else{
                        error += fabs(richardson_convolutions[conv_ind] - 
                                      derivatives[''' + str(nmr_steps - nmr_convolutions + 1) + ''' + conv_ind]) * fact;
                    }
                    
                    errors[conv_ind] = error;
                }
            }
        '''
        return func

    def _richardson_error_kernel(self, nmr_steps, nmr_convolutions, richardson_coefficients):
        func = ''
        func += self._get_error_estimate_functions_cl(nmr_steps, nmr_convolutions, richardson_coefficients)
        func += '''
            void convolute(mot_data_struct* data){
                _apply_richardson_convolution(data->derivatives, data->richardson_convolutions);
                _compute_richardson_errors(data->derivatives, data->richardson_convolutions, data->errors);
            }
        '''
        return SimpleNamedCLFunction(func, 'convolute')

    def _get_single_step_kernel(self, model, nmr_params, nmr_steps, step_ratio):
        func = ''
        func += self._get_compute_functions_cl(model, nmr_params, nmr_steps, step_ratio)
        func += '''
            void compute(mot_data_struct* data){
                mot_float_type x_input[''' + str(nmr_params) + '''];
                
                for(uint param_ind = 0; param_ind < ''' + str(nmr_params) + '''; param_ind++){
                    x_input[param_ind] = data->parameters[param_ind];
                }
                double f_x_input = _calculate_function(data, x_input);
                
                uint param_ind = 0;
                for(uint px = 0; px < ''' + str(nmr_params) + '''; px++){
                    for(uint py = px; py < ''' + str(nmr_params) + '''; py++){
                        _compute_steps(data, x_input, f_x_input, px, py, 
                                       data->step_evaluates + param_ind * ''' + str(nmr_steps) + ''');    
                        param_ind += 1;
                    }
                }
            }
        '''
        return SimpleNamedCLFunction(func, 'compute')


"""
Everything under this comment is copied from numdifftools (https://github.com/pbrod/numdifftools) to remove a dependency.

In the future these extrapolation methods should be translated to OpenCL for fast evaluation over multiple instances.
"""


class Extrapolate(object):

    def __init__(self, step_ratio):
        self._step_ratio = step_ratio

    def __call__(self, el):
        richardson_derivatives, richardson_errors, nmr_params = el

        def _wynn_extrapolate(der):
            der, errors = dea3(der[0:-2], der[1:-1], der[2:], symmetric=False)
            return der, errors

        def _add_error_to_outliers(der, trim_fact=10):
            # discard any estimate that differs wildly from the
            # median of all estimates. A factor of 10 to 1 in either
            # direction is probably wild enough here. The actual
            # trimming factor is defined as a parameter.
            try:
                median = np.nanmedian(der, axis=0)
                p75 = np.nanpercentile(der, 75, axis=0)
                p25 = np.nanpercentile(der, 25, axis=0)
                iqr = np.abs(p75 - p25)
            except ValueError as msg:
                return 0 * der

            a_median = np.abs(median)
            outliers = (((abs(der) < (a_median / trim_fact)) +
                         (abs(der) > (a_median * trim_fact))) * (a_median > 1e-8) +
                        ((der < p25 - 1.5 * iqr) + (p75 + 1.5 * iqr < der)))
            errors = outliers * np.abs(der - median)
            return errors

        def _get_arg_min(errors):
            shape = errors.shape
            try:
                arg_mins = np.nanargmin(errors, axis=0)
                min_errors = np.nanmin(errors, axis=0)
            except ValueError as msg:
                return np.arange(shape[1])

            for i, min_error in enumerate(min_errors):
                idx = np.flatnonzero(errors[:, i] == min_error)
                arg_mins[i] = idx[idx.size // 2]
            return np.ravel_multi_index((arg_mins, np.arange(shape[1])), shape)

        def _get_best_estimate(der, errors):
            errors += _add_error_to_outliers(der)
            ix = _get_arg_min(errors)
            err = errors.flat[ix]
            return der.flat[ix], err

        def _results_vector_to_matrix(vectors, nmr_params):
            matrices = np.zeros(vectors.shape[:-1] + (nmr_params, nmr_params), dtype=vectors.dtype)

            ltr_ind = 0
            for px in range(nmr_params):
                matrices[..., px, px] = vectors[..., ltr_ind]
                ltr_ind += 1

                for py in range(px + 1, nmr_params):
                    matrices[..., px, py] = vectors[..., ltr_ind]
                    matrices[..., py, px] = vectors[..., ltr_ind]
                    ltr_ind += 1
            return matrices

        if len(richardson_derivatives) > 2:
            richardson_derivatives, richardson_errors = _wynn_extrapolate(richardson_derivatives)
        derivatives, err_test = _get_best_estimate(richardson_derivatives, richardson_errors)

        return derivatives


EPS = np.finfo(np.float64).eps
_EPS = EPS
_TINY = np.finfo(np.float64).tiny


def max_abs(a1, a2):
    return np.maximum(np.abs(a1), np.abs(a2))


def dea3(v0, v1, v2, symmetric=False):
    """
    Extrapolate a slowly convergent sequence

    Parameters
    ----------
    v0, v1, v2 : array-like
        3 values of a convergent sequence to extrapolate

    Returns
    -------
    result : array-like
        extrapolated value
    abserr : array-like
        absolute error estimate

    Description
    -----------
    DEA3 attempts to extrapolate nonlinearly to a better estimate
    of the sequence's limiting value, thus improving the rate of
    convergence. The routine is based on the epsilon algorithm of
    P. Wynn, see [1]_.

     Example
     -------
     # integrate sin(x) from 0 to pi/2

     >>> import numpy as np
     >>> import numdifftools as nd
     >>> Ei= np.zeros(3)
     >>> linfun = lambda i : np.linspace(0, np.pi/2., 2**(i+5)+1)
     >>> for k in np.arange(3):
     ...    x = linfun(k)
     ...    Ei[k] = np.trapz(np.sin(x),x)
     >>> [En, err] = nd.dea3(Ei[0], Ei[1], Ei[2])
     >>> truErr = Ei-1.
     >>> (truErr, err, En)
     (array([ -2.00805680e-04,  -5.01999079e-05,  -1.25498825e-05]),
     array([ 0.00020081]), array([ 1.]))

     See also
     --------
     dea

     Reference
     ---------
     .. [1] C. Brezinski and M. Redivo Zaglia (1991)
            "Extrapolation Methods. Theory and Practice", North-Holland.

    ..  [2] C. Brezinski (1977)
            "Acceleration de la convergence en analyse numerique",
            "Lecture Notes in Math.", vol. 584,
            Springer-Verlag, New York, 1977.

    ..  [3] E. J. Weniger (1989)
            "Nonlinear sequence transformations for the acceleration of
            convergence and the summation of divergent series"
            Computer Physics Reports Vol. 10, 189 - 371
            http://arxiv.org/abs/math/0306302v1
    """
    e0, e1, e2 = np.atleast_1d(v0, v1, v2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore division by zero and overflow
        delta2, delta1 = e2 - e1, e1 - e0
        err2, err1 = np.abs(delta2), np.abs(delta1)
        tol2, tol1 = max_abs(e2, e1) * _EPS, max_abs(e1, e0) * _EPS
        delta1[err1 < _TINY] = _TINY
        delta2[err2 < _TINY] = _TINY  # avoid division by zero and overflow
        ss = 1.0 / delta2 - 1.0 / delta1 + _TINY
        smalle2 = abs(ss * e1) <= 1.0e-3
        converged = (err1 <= tol1) & (err2 <= tol2) | smalle2
        result = np.where(converged, e2 * 1.0, e1 + 1.0 / ss)
    abserr = err1 + err2 + np.where(converged, tol2 * 10, np.abs(result - e2))
    if symmetric and len(result) > 1:
        return result[:-1], abserr[1:]
    return result, abserr
