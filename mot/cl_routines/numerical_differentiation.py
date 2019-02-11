from numbers import Number
import numpy as np
from mot.lib.cl_function import SimpleCLFunction
from mot.lib.kernel_data import Array, Zeros, LocalMemory

from mot.library_functions import SimpleCLLibrary

__author__ = 'Robbert Harms'
__date__ = '2017-10-16'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


def estimate_hessian(objective_func, parameters,
                     lower_bounds=None, upper_bounds=None,
                     step_ratio=2, nmr_steps=5,
                     max_step_sizes=None,
                     data=None, cl_runtime_info=None):
    """Estimate and return the upper triangular elements of the Hessian of the given function at the given parameters.

    This calculates the Hessian using central difference (using a 2nd order Taylor expansion) with a Richardson
    extrapolation over the proposed sequence of steps. If enough steps are given, we apply a Wynn epsilon extrapolation
    on top of the Richardson extrapolated results. If more steps are left, we return the estimate with the lowest error,
    taking into account outliers using a median filter.

    The Hessian is evaluated at the steps:

    .. math::
        \quad  ((f(x + d_j e_j + d_k e_k) - f(x + d_j e_j - d_k e_k)) -
                (f(x - d_j e_j + d_k e_k) - f(x - d_j e_j - d_k e_k)) /
                (4 d_j d_k)

    where :math:`e_j` is a vector where element :math:`j` is one and the rest are zero
    and :math:`d_j` is a scalar spacing :math:`steps_j`.

    Steps are generated according to an exponentially diminishing ratio, defined as:

        steps = max_step * step_ratio**-i, i = 0,1,..,nmr_steps-1.

    Where the maximum step can be provided. For example, a maximum step of 2 with a step ratio of 2, computed for
    4 steps gives: [2.0, 1.0, 0.5, 0.25]. If lower and upper bounds are given, we use as maximum step size the largest
    step size that fits between the Hessian point and the boundaries.

    The steps define the order of the estimation, with 2 steps resulting in a O(h^2) estimate, 3 steps resulting in a
    O(h^4) estimate and 4 or more steps resulting in a O(h^6) derivative estimate.

    Args:
        objective_func (mot.lib.cl_function.CLFunction): The function we want to differentiate.
            A CL function with the signature:

            .. code-block:: c

                double <func_name>(local const mot_float_type* const x, void* data);

            The objective function has the same signature as the minimization function in MOT. For the numerical
            hessian, the ``objective_list`` parameter is ignored.

        parameters (ndarray): The parameters at which to evaluate the gradient. A (d, p) matrix with d problems,
            and p parameters
        lower_bounds (list or None): a list of length (p,) for p parameters with the lower bounds.
            Each element of the list can be a scalar or a vector (of the same length as the number
            of problem instances). To disable bounds for this parameter use -np.inf.
        upper_bounds (list or None): a list of length (p,) for p parameters with the upper bounds.
            Each element of the list can be a scalar or a vector (of the same length as the number
            of problem instances). To disable bounds for this parameter use np.inf.
        step_ratio (float): the ratio at which the steps diminish.
        nmr_steps (int): the number of steps we will generate. We will calculate the derivative for each of these
            step sizes and extrapolate the best step size from among them. The minimum number of steps is 1.
        max_step_sizes (float or ndarray or None): the maximum step size, or the maximum step size per parameter.
            If None is given, we use 0.1 for all parameters. If a float is given, we use that for all parameters.
            If a list is given, it should be of the same length as the number of parameters.
        data (mot.lib.kernel_data.KernelData): the user provided data for the ``void* data`` pointer.
        cl_runtime_info (mot.configuration.CLRuntimeInfo): the runtime information

    Returns:
        ndarray: per problem instance a vector with the upper triangular elements of the Hessian matrix.
            This array can hold NaN's, for elements where the Hessian failed to approximate.
    """
    if len(parameters.shape) == 1:
        parameters = parameters[None, :]

    nmr_voxels = parameters.shape[0]
    nmr_params = parameters.shape[1]
    nmr_derivatives = nmr_params * (nmr_params + 1) // 2

    initial_step = _get_initial_step(parameters, lower_bounds, upper_bounds, max_step_sizes)

    kernel_data = {
        'parameters': Array(parameters, ctype='mot_float_type'),
        'initial_step': Array(initial_step, ctype='float'),
        'derivatives': Zeros((nmr_voxels, nmr_derivatives), 'double'),
        'errors': Zeros((nmr_voxels, nmr_derivatives), 'double'),
        'x_tmp': LocalMemory('mot_float_type', nmr_params),
        'data': data,
        'scratch': LocalMemory('double', nmr_steps + (nmr_steps - 1) + nmr_steps)
    }

    hessian_kernel = SimpleCLFunction.from_string('''
        void _numdiff_hessian(
                global mot_float_type* parameters,
                global float* initial_step,
                global double* derivatives,
                global double* errors,
                local mot_float_type* x_tmp,
                void* data,
                local double* scratch){

            if(get_local_id(0) == 0){
                for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                    x_tmp[i] = parameters[i];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            double f_x_input = ''' + objective_func.get_cl_function_name() + '''(x_tmp, data);

            // upper triangle loop
            uint coord_ind = 0;
            for(int i = 0; i < ''' + str(nmr_params) + '''; i++){
                for(int j = i; j < ''' + str(nmr_params) + '''; j++){
                    _numdiff_hessian_element(
                        data, x_tmp, f_x_input, i, j, initial_step, 
                        derivatives + coord_ind, errors + coord_ind, scratch);

                    coord_ind++;
                }
            }
        }
    ''', dependencies=[objective_func,
                       _get_numdiff_hessian_element_func(objective_func, nmr_steps, step_ratio)])

    hessian_kernel.evaluate(kernel_data, nmr_voxels, use_local_reduction=True, cl_runtime_info=cl_runtime_info)

    return kernel_data['derivatives'].get_data()


def _get_numdiff_hessian_element_func(objective_func, nmr_steps, step_ratio):
    """Return a function to compute one element of the Hessian matrix."""
    return SimpleCLFunction.from_string('''
        /**
         * Compute the Hessian using (possibly) multiple steps with various interpolations. 
         */ 
        void _numdiff_hessian_element(
                void* data, local mot_float_type* x_tmp, mot_float_type f_x_input,
                uint px, uint py, global float* initial_step, global double* derivative, 
                global double* error, local double* scratch){

            const uint nmr_steps = ''' + str(nmr_steps) + ''';
            uint nmr_steps_remaining = nmr_steps;

            local double* scratch_ind = scratch;
            local double* steps = scratch_ind;      scratch_ind += nmr_steps; 
            local double* errors = scratch_ind;     scratch_ind += nmr_steps - 1;
            local double* steps_tmp = scratch_ind;  scratch_ind += nmr_steps;
            
            if(get_local_id(0) == 0){
                for(int i = 0; i < nmr_steps - 1; i++){
                    errors[i] = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            _numdiff_hessian_steps(data, x_tmp, f_x_input, px, py, steps, initial_step);

            if(nmr_steps_remaining > 1){
                nmr_steps_remaining = _numdiff_hessian_richardson_extrapolation(steps); 
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            if(nmr_steps_remaining >= 3){
                nmr_steps_remaining = _numdiff_wynn_extrapolation(steps, errors, nmr_steps_remaining);
                barrier(CLK_LOCAL_MEM_FENCE);                
            }
            
            if(nmr_steps_remaining > 1){
                _numdiff_find_best_step(steps, errors, steps_tmp, nmr_steps_remaining);
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            if(get_local_id(0) == 0){
                *derivative = steps[0];
                *error = errors[0];
            }
        }
    ''', dependencies=[
        _get_numdiff_hessian_steps_func(objective_func, nmr_steps, step_ratio),
        _get_numdiff_hessian_richardson_extrapolation_func(nmr_steps, step_ratio),
        _get_numdiff_wynn_extrapolation_func(),
        _get_numdiff_find_best_step_func()
    ])


def _get_numdiff_hessian_steps_func(objective_func, nmr_steps, step_ratio):
    """Get a function to compute the multiple step sizes for a single element of the Hessian."""
    return SimpleCLFunction.from_string('''
        /**
         * Compute one element of the Hessian for a number of steps.
         * 
         * This uses the initial steps in the data structure, indexed by the parameters to change (px, py).
         *
         * Args:
         *  data: the data container
         *  x_tmp: the array with the input parameters, needs to be writable, although it will return
         *         the same values.
         *  f_x_input: the objective function value at the original set of parameters  
         *  px: the index of the first parameter to perturbate
         *  py: the index of the second parameter to perturbate
         *  steps: storage location for the output steps
         *  initial_step: the initial steps, array of same length as x_temp
         */
        void _numdiff_hessian_steps(void* data, local mot_float_type* x_tmp, 
                                    mot_float_type f_x_input,
                                    uint px, uint py, 
                                    local double* steps, 
                                    global float* initial_step){

            double step_x;
            double step_y;
            double tmp;
            bool is_first_workitem = get_local_id(0) == 0;

            if(px == py){
                for(uint step_ind = 0; step_ind < ''' + str(nmr_steps) + '''; step_ind++){
                    step_x = initial_step[px] / pown(''' + str(float(step_ratio)) + ''', step_ind);

                    tmp = (
                          _numdiff_hessian_eval_step_mono(data, x_tmp, px, 2 * step_x)
                        + _numdiff_hessian_eval_step_mono(data, x_tmp, px, -2 * step_x)
                        - 2 * f_x_input
                    ) / (4 * step_x * step_x);    

                    if(is_first_workitem){
                        steps[step_ind] = tmp;
                    }
                }
            }
            else{
                for(uint step_ind = 0; step_ind < ''' + str(nmr_steps) + '''; step_ind++){
                    step_x = initial_step[px] / pown(''' + str(float(step_ratio)) + ''', step_ind);
                    step_y = initial_step[py] / pown(''' + str(float(step_ratio)) + ''', step_ind);

                    tmp = (
                          _numdiff_hessian_eval_step_bi(data, x_tmp, px, step_x, py, step_y)
                        - _numdiff_hessian_eval_step_bi(data, x_tmp, px, step_x, py, -step_y)
                        - _numdiff_hessian_eval_step_bi(data, x_tmp, px, -step_x, py, step_y)
                        + _numdiff_hessian_eval_step_bi(data, x_tmp, px, -step_x, py, -step_y)
                    ) / (4 * step_x * step_y);

                    if(is_first_workitem){
                        steps[step_ind] = tmp;
                    }                       
                }
            }
        }
    ''', dependencies=[SimpleCLFunction.from_string('''
        /**
         * Evaluate the model with a perturbation in one dimensions.
         *
         * Args:
         *  data: the data container
         *  x_tmp: the array with the input parameters, needs to be writable, although it will return
         *         the same values.
         *  perturb_dim0: the index (into the x_tmp parameters) of the parameter to perturbate
         *  perturb_0: the added perturbation of the index corresponding to ``perturb_dim_0``
         *
         * Returns:
         *  the function evaluated at the parameters plus their perturbation.
         */
        double _numdiff_hessian_eval_step_mono(
                void* data, local mot_float_type* x_tmp, 
                uint perturb_dim_0, mot_float_type perturb_0){

            mot_float_type old_0;
            double return_val;

            if(get_local_id(0) == 0){
                old_0 = x_tmp[perturb_dim_0];
                x_tmp[perturb_dim_0] += perturb_0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            return_val = ''' + objective_func.get_cl_function_name() + '''(x_tmp, data);
            barrier(CLK_LOCAL_MEM_FENCE);

            if(get_local_id(0) == 0){
                x_tmp[perturb_dim_0] = old_0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            return return_val;
        }
    '''), SimpleCLFunction.from_string('''
        /**
         * Evaluate the model with a perturbation in two dimensions.
         *
         * Args:
         *  data: the data container
         *  x_tmp: the array with the input parameters, needs to be writable, although it will return
         *         the same values.
         *  perturb_dim_0: the index (into the x_tmp parameters) of the first parameter to perturbate
         *  perturb_0: the added perturbation of the index corresponding to ``perturb_dim_0``
         *  perturb_dim_1: the index (into the x_tmp parameters) of the second parameter to perturbate
         *  perturb_1: the added perturbation of the index corresponding to ``perturb_dim_1``
         *
         * Returns:
         *  the function evaluated at the parameters plus their perturbation.
         */
        double _numdiff_hessian_eval_step_bi(
                void* data, local mot_float_type* x_tmp, 
                uint perturb_dim_0, mot_float_type perturb_0,
                uint perturb_dim_1, mot_float_type perturb_1){

            mot_float_type old_0;
            mot_float_type old_1;
            double return_val;

            if(get_local_id(0) == 0){
                old_0 = x_tmp[perturb_dim_0];
                old_1 = x_tmp[perturb_dim_1];

                x_tmp[perturb_dim_0] += perturb_0;
                x_tmp[perturb_dim_1] += perturb_1;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            return_val = ''' + objective_func.get_cl_function_name() + '''(x_tmp, data);
            barrier(CLK_LOCAL_MEM_FENCE);

            if(get_local_id(0) == 0){
                x_tmp[perturb_dim_0] = old_0;
                x_tmp[perturb_dim_1] = old_1;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            return return_val;
        }
    ''')])


def _get_numdiff_hessian_richardson_extrapolation_func(nmr_steps, step_ratio):
    return SimpleCLFunction.from_string('''
        /**
         * Apply the Richardson extrapolation to the derivatives computed with multiple steps.
         * 
         * Having for every problem instance and every Hessian element multiple derivatives computed 
         * with decreasing steps, we can now apply Richardson extrapolation to reduce the error term from O(h^2) to
         * O(h^4) or O(h^6) depending on how many steps we have calculated.
         *
         * This method only considers extrapolation up to the sixth error order. For a set of two derivatives we compute
         * a single fourth order approximation, for three derivatives or more, we compute ``n-2`` sixth order 
         * approximations. 
         * 
         * Args:
         *  steps: on input, the steps we are convoluting with Richardson extrapolation. On output, 
         *         the convoluted steps, these are less than the input, see the return value.
         * 
         * Returns:
         *  nmr_steps: the number of steps remaining after convolution
         */
        uint _numdiff_hessian_richardson_extrapolation(local double* steps){
            
            const uint nmr_steps = ''' + str(nmr_steps) + ''';
            uint nmr_steps_remaining = nmr_steps;
            
            if(get_local_id(0) == 0){
                // 4th order approximations
                for(uint i = 0; i < nmr_steps - 1; i++){
                    steps[i] = richardson_extrapolate(steps[i], steps[i + 1], ''' + str(step_ratio) + ''', 2);
                }
                nmr_steps_remaining--;
                
                // 6th order approximations
                for(uint i = 0; i < nmr_steps - 2; i++){
                    steps[i] = richardson_extrapolate(steps[i], steps[i + 1], ''' + str(step_ratio) + ''', 4);
                }
                nmr_steps_remaining--;
            }

            return nmr_steps_remaining;
        }
    ''', dependencies=[richardson_extrapolation()])


def _get_numdiff_find_best_step_func():
    return SimpleCLFunction.from_string('''
        /**
         * Find the step with the lowest error.
         *
         * This will first apply a median filter on the steps to locate derivatives which deviate strongly 
         * from the distribution of the steps. These deviations are added as extra error term to the error term from 
         * the extrapolations. Afterwards it chooses the derivative with the lowest error. 
         *
         * Args:
         *  steps: on input, array with the input steps. On output, a single step with the best derivative
         *  errors: on input, array with the error terms per step, on output the error for the returned step
         *  scratch: temporary array of the same length as the steps array
         *  nmr_steps: the number of steps in the input array
         */
        void _numdiff_find_best_step(local double* steps, local double* errors, local double* scratch, uint nmr_steps){
            if(get_local_id(0) == 0){
                bool all_nan = true;
                for(int i = 0; i < nmr_steps; i++){
                    if(isnan(steps[i])){
                        steps[i] = 0;
                    }
                    else{
                        all_nan = false;
                    }
                }
    
                if(all_nan){
                    steps[0] = NAN;
                    errors[0] = NAN;
    
                    return;
                }
    
                _get_median_outlier_error(steps, scratch, nmr_steps);
                
                double lowest_error = INFINITY;
                int ind_lowest_error = 0; 
                for(int i = 0; i < nmr_steps; i++){
                    if(errors[i] + scratch[i] < lowest_error){
                        lowest_error = errors[i] + scratch[i];
                        ind_lowest_error = i;
                    } 
                }
    
                steps[0] = steps[ind_lowest_error];
                errors[0] = errors[ind_lowest_error] + scratch[ind_lowest_error];    
            }
        }
    ''', dependencies=[SimpleCLFunction.from_string('''
        /**
         * This function tries to detect outliers by means of an median filter. 
         * If a value is a factor of 10 to 1 in either direction away from the median, we weight it 
         * with an additional error.
         *
         * This uses a hardcoded trim factor of 10 to 1 as relative distance.
         * 
         * Args:
         *  steps: 
         * 
         */
        void _get_median_outlier_error(local double* steps, local double* median_errors, uint nmr_steps){
            float trim_factor = 10;   
            int i;

            local double* sorted_values = median_errors;
            for(int i = 0; i < nmr_steps; i++){
                sorted_values[i] = steps[i];
            }            
            _numdiff_sort_values(sorted_values, nmr_steps);

            double p25 = _numdiff_get_percentile(sorted_values, nmr_steps, 0.25);
            double p50 = _numdiff_get_percentile(sorted_values, nmr_steps, 0.50);
            double p75 = _numdiff_get_percentile(sorted_values, nmr_steps, 0.75);

            double iqr = fabs(p75 - p25);
            double abs_median = fabs(p50);

            bool is_outlier;
            for(i = 0; i < nmr_steps; i++){
                is_outlier = (((fabs(steps[i]) < (abs_median / trim_factor)) +
                               (fabs(steps[i]) > (abs_median * trim_factor))) * (abs_median > 1e-8) +
                              ((steps[i] < p25 - 1.5 * iqr) + (p75 + 1.5 * iqr < steps[i])));

                median_errors[i] = is_outlier * fabs(steps[i] - p50);
            }            
        }

    ''', dependencies=[SimpleCLFunction.from_string('''
            /** 
             * Uses bubblesort to sort the given values in ascending order.
             * 
             */
            void _numdiff_sort_values(local double* values, int n){  
                int i, j;
                double tmp;

                for(i = 0; i < n; i++){
                    for(j = 0; j < n - i - 1; j++){
                        if(values[j] > values[j + 1]){
                            tmp = values[j];
                            values[j] = values[j + 1];
                            values[j + 1] = tmp;    
                        }
                    }
                }
            }
        '''), SimpleCLFunction.from_string('''
            /**
             * Find the distribution value for the requested percentile.
             * 
             * This uses linear interpolation with C == 1, as defined in 
             * https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method.
             * 
             * Args:
             *  sorted_values: array of sorted values
             *  nmr_values: number of values
             *  percentile: the requested percentile, between 0 and 1.
             * 
             * Returns:
             *  the value for the percentile
             */
            double _numdiff_get_percentile(local double* sorted_values, int nmr_values, double percentile){
                double rank = percentile * (nmr_values - 1);
                int rank_int = (int) rank;
                double fraction = rank - rank_int;
                return sorted_values[rank_int] + fraction * (sorted_values[rank_int + 1] - sorted_values[rank_int]);    
            }
        ''')])])


def _get_numdiff_wynn_extrapolation_func():
    return SimpleCLFunction.from_string('''
        /**
         * Apply Wynn extrapolation to the derivatives.
         *
         * This algorithm, known in the Python Numdifftools as DEA3, attempts to extrapolate non-linearly to a better
         * estimate of the sequence's limiting value, thus improving the rate of convergence. The routine is based on the
         * epsilon algorithm of P. Wynn [1].
         *
         * References:
         *   - [1] C. Brezinski and M. Redivo Zaglia (1991)
         *           "Extrapolation Methods. Theory and Practice", North-Holland.
         *
         *   - [2] C. Brezinski (1977)
         *           "Acceleration de la convergence en analyse numerique",
         *           "Lecture Notes in Math.", vol. 584,
         *           Springer-Verlag, New York, 1977.
         * 
         *   - [3] E. J. Weniger (1989)
         *           "Nonlinear sequence transformations for the acceleration of
         *           convergence and the summation of divergent series"
         *           Computer Physics Reports Vol. 10, 189 - 371
         *           http://arxiv.org/abs/math/0306302v1
         * 
         * Args:
         *  steps: on input, array with the input steps. On output, the extrapolated steps
         *  errors: the errors of the extrapolated steps
         *  nmr_steps: the number of steps in the input array
         * 
         * Returns:
         *  the number of steps in the output 
         */
        uint _numdiff_wynn_extrapolation(local double* steps, local double* errors, uint nmr_steps){
            if(get_local_id(0) == 0){
                double v0, v1, v2; 

                double delta0, delta1; 
                double err0, err1; 
                double tol0, tol1; 

                double ss;
                bool converged;

                double result;

                for(uint i = 0; i < nmr_steps - 2; i++){
                    v0 = steps[i];
                    v1 = steps[i + 1];
                    v2 = steps[i + 2];

                    delta0 = v1 - v0;
                    delta1 = v2 - v1;

                    err0 = fabs(delta0);
                    err1 = fabs(delta1);

                    tol0 = max(fabs(v0), fabs(v1)) * MOT_EPSILON;
                    tol1 = max(fabs(v1), fabs(v2)) * MOT_EPSILON;

                    // avoid division by zero and overflow
                    if(err0 < MOT_MIN){
                        delta0 = MOT_MIN;
                    }
                    if(err1 < MOT_MIN){
                        delta1 = MOT_MIN;
                    }

                    ss = 1.0 / delta1 - 1.0 / delta0 + MOT_MIN;
                    converged = ((err0 <= tol0) && (err1 <= tol1)) || (fabs(ss * v1) <= 1e-3);

                    result = (converged ? v2 : v1 + 1.0/ss);

                    steps[i] = result;
                    errors[i] = err0 + err1 + (converged ? tol1 * 10 : fabs(result - v2));
                }
            }
            return nmr_steps - 2;
        }
    ''', dependencies=[])


def _get_initial_step(parameters, lower_bounds, upper_bounds, max_step_sizes):
    """Get an initial step size to use for every parameter.

    This chooses the step sizes based on the maximum step size and the lower and upper bounds.

    Args:
        parameters (ndarray): The parameters at which to evaluate the gradient. A (d, p) matrix with d problems,
            p parameters and n samples.
        lower_bounds (list): lower bounds
        upper_bounds (list): upper bounds
        max_step_sizes (list or None): the maximum step size, or the maximum step size per parameter. Defaults to 0.1

    Returns:
        ndarray: for every problem instance the vector with the initial step size for each parameter.
    """
    nmr_params = parameters.shape[1]

    initial_step = np.zeros_like(parameters)

    if max_step_sizes is None:
        max_step_sizes = 0.1
    if isinstance(max_step_sizes, Number):
        max_step_sizes = [max_step_sizes] * nmr_params
    max_step_sizes = np.array(max_step_sizes)

    for ind in range(parameters.shape[1]):
        minimum_allowed_step = np.minimum(np.abs(parameters[:, ind] - lower_bounds[ind]),
                                          np.abs(upper_bounds[ind] - parameters[:, ind]))
        initial_step[:, ind] = np.minimum(minimum_allowed_step, max_step_sizes[ind])

    return initial_step / 2.


class richardson_extrapolation(SimpleCLLibrary):

    def __init__(self):
        """Get the Richardson extrapolation function."""
        super().__init__('''
            /**
             * Apply a Richardson extrapolation on the given input.
             *
             * See https://en.wikipedia.org/wiki/Richardson_extrapolation. 
             *
             * Args:
             *  v1: the primary value at a step size of h
             *  v2: the secondary value, computed at a step size of h/step_ratio 
             *  step_ratio: the ratio at which the steps diminish
             *  approximation_order: the order of the current approximations
             * 
             * Returns:
             *   the Richardson extrapolation
             */
            double richardson_extrapolate(double v1, double v2, float step_ratio, uint approximation_order){
                double div = 1 / (pown(step_ratio, approximation_order) - 1);
                return (div + 1) * v2 - div * v1;
            }
        ''')
