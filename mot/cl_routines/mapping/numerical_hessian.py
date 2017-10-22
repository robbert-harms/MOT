import multiprocessing
import numpy as np
import os
from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import KernelInputBuffer, SimpleNamedCLFunction, KernelInputLocalMemory
from ...cl_routines.base import CLRoutine
from scipy import linalg
import warnings


__author__ = 'Robbert Harms'
__date__ = '2017-10-16'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class NumericalHessian(CLRoutine):

    def calculate(self, cl_func, kernel_data, nmr_inst_per_problem, parameters, step_generator=None,
                  codec=None, double_precision=False, parameter_scalings=None):
        """Calculate and return the Hessian of the given function at the given parameters.

        This calculates the Hessian using a central difference at a 2nd order Taylor expansion.

        .. math::
            \quad  ((f(x + d_j e_j + d_k e_k) - f(x + d_j e_j - d_k e_k)) -
                    (f(x - d_j e_j + d_k e_k) - f(x - d_j e_j - d_k e_k)) /
                    (4 d_j d_k)

        where :math:`e_j` is a vector where element :math:`j` is one and the rest are zero
        and :math:`d_j` is a scalar spacing :math:`steps_j`.

        This method evaluates the Hessian at various step sizes and uses the Richardson extrapolation method
        to approximate the final Hessian.

        This code is a CL accelerated version of the default Hessian calculation method of the NumDiffTool package
        (https://github.com/pbrod/numdifftools).

        Args:
            cl_func (mot.utils.NamedCLFunction): The function we would like to compute the gradient of.
                This CL function should accept the arguments "data, x, observation_ind".
            kernel_data (dict): the dictionary with the extra kernel data to load
            nmr_inst_per_problem (int): the number of instances to evaluate per problem
            parameters (ndarray): The parameters at which to evaluate the gradient. A (d, p) matrix with d problems,
                p parameters and n samples.
            step_generator (StepGenerator): a generator for generating steps for each parameters with a
                decreasing order of magnitude. If not given we assume a initial step size of 1 in each direction.
            codec (mot.model_building.utils.ParameterCodec): a parameter codec used to constrain the parameters
                within bounds.
            double_precision (boolean): if we want to calculate everything in double precision
            parameter_scalings (ndarray): a 1d array with parameter scaling factors to ensure all parameters
                are approximately in the same order of magnitude. This should return the values such that when
                the parameters are multiplied with this scaling all parameters are in the same order of magnitude.

        Returns:
            ndarray: the gradients for each of the parameters for each of the problems
        """
        np_dtype = np.float32
        if double_precision:
            np_dtype = np.float64

        if len(parameters.shape) == 1:
            parameters = parameters[None, :]
        nmr_params = parameters.shape[1]

        if step_generator is None:
            step_generator = SimpleStepGenerator(np.ones(nmr_params))
        steps = step_generator.generate_steps().astype(np_dtype)

        if parameter_scalings is None:
            parameter_scalings = np.ones(nmr_params, dtype=np.float64)
        parameter_scalings = np.asarray(parameter_scalings, dtype=np.float64)

        elements_needed = (nmr_params**2 - nmr_params) // 2 + nmr_params
        step_evaluates = np.zeros((parameters.shape[0], steps.shape[0], elements_needed), dtype=np.float64)
        step_evaluates_convoluted = np.zeros((parameters.shape[0], steps.shape[0], elements_needed), dtype=np.float64)

        all_kernel_data = dict(kernel_data)
        all_kernel_data.update({
            'parameters': KernelInputBuffer(parameters),
            'local_reduction_lls': KernelInputLocalMemory(np.float64),
            'parameter_scalings': KernelInputBuffer(parameter_scalings, offset_str='0'),
            'steps': KernelInputBuffer(steps, offset_str='0'),
            'step_evaluates': KernelInputBuffer(step_evaluates, is_writable=True),
            'step_evaluates_convoluted': KernelInputBuffer(step_evaluates_convoluted, is_writable=True),
        })

        import time
        start = time.time()
        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._get_wrapped_function(cl_func, parameters, nmr_inst_per_problem, codec, steps),
                             all_kernel_data, parameters.shape[0], double_precision=double_precision,
                             use_local_reduction=True)

        print(time.time() - start)

        step_evaluates = all_kernel_data['step_evaluates'].get_data()
        step_evaluates_convoluted = all_kernel_data['step_evaluates_convoluted'].get_data()

        full_step_evaluates = np.zeros((parameters.shape[0], steps.shape[0], nmr_params, nmr_params), dtype=np.float64)
        full_step_evaluates_convoluted = np.zeros((parameters.shape[0], steps.shape[0], nmr_params, nmr_params), dtype=np.float64)

        ltr_ind = 0
        for px in range(nmr_params):
            full_step_evaluates[..., px, px] = step_evaluates[..., ltr_ind]
            full_step_evaluates_convoluted[..., px, px] = step_evaluates_convoluted[..., ltr_ind]
            ltr_ind += 1

            for py in range(px + 1, nmr_params):
                full_step_evaluates[..., px, py] = step_evaluates[..., ltr_ind]
                full_step_evaluates_convoluted[..., px, py] = step_evaluates_convoluted[..., ltr_ind]

                full_step_evaluates[..., py, px] = step_evaluates[..., ltr_ind]
                full_step_evaluates_convoluted[..., py, px] = step_evaluates_convoluted[..., ltr_ind]

                ltr_ind += 1

        extrapolate = Extrapolate(steps, step_generator.get_step_ratio())

        def data_iterator():
            for param_ind in range(parameters.shape[0]):
                yield (full_step_evaluates[param_ind], full_step_evaluates_convoluted[param_ind])

        if os.name == 'nt':  # In Windows there is no fork.
            return np.array(list(map(extrapolate, data_iterator())))
        else:
            try:
                p = multiprocessing.Pool()
                return_data = np.array(list(p.imap(extrapolate, data_iterator())))
                p.close()
                p.join()
                return return_data
            except OSError:
                return np.array(list(map(extrapolate, data_iterator())))

    def _get_wrapped_function(self, ll_function, parameters, nmr_inst_per_problem, codec, steps):
        nmr_params = parameters.shape[1]
        elements_needed = (nmr_params ** 2 - nmr_params) // 2 + nmr_params

        func = ''

        if codec is not None:
            func += codec.get_parameter_decode_function(function_name='decode')
            func += codec.get_parameter_encode_function(function_name='encode')

        func += ll_function.get_cl_code()
        func += '''
            double calculate_function(mot_data_struct* data, mot_float_type* x){
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

            void apply_bounds(mot_data_struct* data, mot_float_type* params){
        '''
        if codec is not None:
            func += '''
                encode(data, params);
                decode(data, params);
            '''
        func += '''
            }
            
            double eval_step(mot_data_struct* data, mot_float_type* x_input, 
                             uint perturb_loc_0, mot_float_type perturb_0,
                             uint perturb_loc_1, mot_float_type perturb_1){
            
                mot_float_type x_tmp[''' + str(nmr_params) + '''];
                for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                    x_tmp[i] = x_input[i];
                }
                x_tmp[perturb_loc_0] += perturb_0;
                x_tmp[perturb_loc_1] += perturb_1;
                
                apply_bounds(data, x_tmp);
                return calculate_function(data, x_tmp);                
            }
            
            /** 
             * Compute the Hessian for one row of step sizes.
             *
             * This method uses a central difference method at the 2nd order Taylor expansion. 
             */
            void compute_step(mot_data_struct* data, mot_float_type* x_input, mot_float_type f_x_input, 
                              global mot_float_type* step_sizes, global double* step_evaluate_ptr){
                
                mot_float_type perturbation[''' + str(nmr_params) + '''];
                double calc;
                uint result_ind = 0;
                
                for(uint px = 0; px < ''' + str(nmr_params) + '''; px++){
                    calc = (
                          eval_step(data, x_input, 
                                    px, 2 * (step_sizes[px] / data->parameter_scalings[px]),
                                    0, 0)
                        + eval_step(data, x_input, 
                                    px, -2 * (step_sizes[px] / data->parameter_scalings[px]),
                                    0, 0)
                        - 2 * f_x_input
                        ) / (4 * step_sizes[px] * step_sizes[px]);
                    
                    if(get_local_id(0) == 0){
                        step_evaluate_ptr[result_ind++] = calc;
                    }
                    
                    for(uint py = px + 1; py < ''' + str(nmr_params) + '''; py++){
                        calc = (
                              eval_step(data, x_input, 
                                        px, step_sizes[px] / data->parameter_scalings[px],
                                        py, step_sizes[py] / data->parameter_scalings[py])
                            - eval_step(data, x_input, 
                                        px, step_sizes[px] / data->parameter_scalings[px],
                                        py, -step_sizes[py] / data->parameter_scalings[py])
                            - eval_step(data, x_input, 
                                        px, -step_sizes[px] / data->parameter_scalings[px],
                                        py, step_sizes[py] / data->parameter_scalings[py])
                            + eval_step(data, x_input, 
                                        px, -step_sizes[px] / data->parameter_scalings[px],
                                        py, -step_sizes[py] / data->parameter_scalings[py])
                        ) / (4 * step_sizes[px] * step_sizes[py]);

                        if(get_local_id(0) == 0){
                            step_evaluate_ptr[result_ind++] = calc;
                        }
                    }
                }    
            }
            
            /**
             * Apply a simple kernel convolution over the results from each row of steps.
             * 
             * This applies a convolution starting from the starting step index that contained a valid move. 
             * It uses the mode 'reflect' to deal with outside points.
             * 
             * Please note that this kernel is hard coded to work with 2nd order Taylor expansions derivatives only.
             
             * Args:
             *  step_evaluates: the step evaluates for each row of the step sizes
             *  step_evaluates_convoluted: the array to place the convoluted results in
             *  start_step_ind: the index of the first step that contains a valid step size (step within bounds for  
             *      every parameter) 
             */ 
            void apply_richardson_convolution(
                    global double* step_evaluates,
                    global double* step_evaluates_convoluted,
                    uint start_step_ind){
                    
                double kernel_2[2] = {-1/3., 1 + 1/3.};
                double kernel_3[3] = {1/45., -20/45., 1 + 20/45. - 1/45.};
                
                double* kernel_ptr;
                uint kernel_length;
                
                if(''' + str(steps.shape[0]) + ''' - start_step_ind <= 1){
                    return;
                }
                
                if(''' + str(steps.shape[0]) + ''' - start_step_ind == 2){
                    kernel_ptr = kernel_2;
                    kernel_length = 2;
                }
                else{
                    kernel_ptr = kernel_3;
                    kernel_length = 3;
                }
                
                uint kernel_step_ind;
                ulong local_id = get_local_id(0);
                uint workgroup_size = get_local_size(0);
                uint element_ind;
                uint elements_for_workitem = ceil(''' + str(elements_needed) + ''' / (mot_float_type)workgroup_size);
                
                if(workgroup_size * (elements_for_workitem - 1) + local_id >= ''' + str(elements_needed) + '''){
                    elements_for_workitem -= 1;
                }
                
                for(uint i = 0; i < elements_for_workitem; i++){
                    element_ind = i * workgroup_size + local_id;        
                    
                    for(uint step_ind = start_step_ind; step_ind < ''' + str(steps.shape[0]) + '''; step_ind++){
                        
                        // convolute kernel
                        for(uint kernel_ind = 0; kernel_ind < kernel_length; kernel_ind++){
                            kernel_step_ind = step_ind + kernel_ind;
                            
                            // reflect
                            if(kernel_step_ind >= ''' + str(steps.shape[0]) + '''){
                                kernel_step_ind -= 2 * (kernel_step_ind - ''' + str(steps.shape[0]) + ''') + 1;  
                            }
                         
                            step_evaluates_convoluted[step_ind * ''' + str(elements_needed) + ''' + element_ind] += 
                                (step_evaluates[kernel_step_ind * ''' + str(elements_needed) + ''' + element_ind] 
                                    * kernel_ptr[kernel_ind]);
                        }
                    }
                }
            }
            
            void compute(mot_data_struct* data){
                uint param_ind;
                double eval;
                double f_x_input;

                mot_float_type x_input[''' + str(nmr_params) + '''];
                global double* step_evaluate_ptr;
                global double* step_evaluate_convoluted_ptr;
                global mot_float_type* parameter_steps_ptr;
                
                for(param_ind = 0; param_ind < ''' + str(nmr_params) + '''; param_ind++){
                    x_input[param_ind] = data->parameters[param_ind];
                }
                f_x_input = calculate_function(data, x_input);

                for(uint step_ind = 0; step_ind < ''' + str(steps.shape[0]) + '''; step_ind++){ 
                    parameter_steps_ptr = data->steps + step_ind * ''' + str(nmr_params) + ''';
                    step_evaluate_ptr = data->step_evaluates + step_ind * ''' + str(elements_needed) + ''';
                    step_evaluate_convoluted_ptr = data->step_evaluates_convoluted + step_ind * ''' + str(elements_needed) + ''';
                    
                    compute_step(data, x_input, f_x_input, parameter_steps_ptr, step_evaluate_ptr);   
                }
                
                apply_richardson_convolution(data->step_evaluates, data->step_evaluates_convoluted, 0);
            }
        '''
        return SimpleNamedCLFunction(func, 'compute')


class StepGenerator(object):
    """A base class for step generators.

    These step generators should return multiple (diminishing) steps at which to evaluate the derivative. The estimates
    at these steps are then extrapolated to estimate the final derivatives.
    """

    def generate_steps(self):
        """Generates a sequence of steps of decreasing magnitude.

        This has to generate steps of logarithmic increase or decrease in order for the later extrapolation to work.

        Please also note that every parameter must diminish in the exact same way

        Returns:
            ndarray: a two matrix with various step sizes (rows) for each parameter (columns).
        """
        raise NotImplementedError()

    def get_step_ratio(self):
        """Get the ratio at which the steps diminish.

        Returns:
            float: the ratio at which the steps diminish, must be larger than 1.
        """
        raise NotImplementedError()


class SimpleStepGenerator(StepGenerator):

    def __init__(self, base_step, step_ratio=2, nmr_steps=15):
        """Generate multiple decreasing steps for derivative calculation.

        This generates steps according to the rule:

            steps = base_step * step_ratio**-i, i=0, 1,.., nmr_steps-1.

        For example, a base step of 2 with a step ratio of 2 and with 4 steps
        results in the step sizes: [2.0, 1.0, 0.5, 0.25]

        A heuristic for the base step is max(log(1 + abs(x)), 1) where x is the mean of a parameter.

        Args:
            base_step (ndarray): Defines the start step, i.e., maximum step per parameter.
            step_ratio (float): The ratio between sequential steps, needs to be larger > 1.
            nmr_steps (int): the number of steps to generate.
        """
        self.base_step = np.asanyarray(base_step, dtype=np.float64)
        self.step_ratio = float(step_ratio)
        self.nmr_steps = nmr_steps

    def generate_steps(self):
        return np.tile(self.base_step, (self.nmr_steps, 1)) * (self.step_ratio ** -np.arange(self.nmr_steps))[:, None]

    def get_step_ratio(self):
        return self.step_ratio


"""
Everything under this comment is copied from numdifftools (https://github.com/pbrod/numdifftools) to remove a dependency.

In the future these extrapolation methods should be translated to OpenCL for fast evaluation over multiple instances.
"""


class Extrapolate(object):

    def __init__(self, steps, step_ratio):
        self._steps = steps
        self._step_ratio = step_ratio

    def __call__(self, el):
        step_evaluates, step_evaluates_convoluted = el
        richardson = Richardson(step_ratio=self._step_ratio, step=2, order=2, num_terms=2)

        def _vstack(sequence, steps):
            original_shape = np.shape(sequence[0])
            f_del = np.vstack(list(np.ravel(r)) for r in sequence)
            h = np.vstack(list(np.ravel(np.ones(original_shape) * step))
                          for step in steps)
            return f_del, h, original_shape

        def _wynn_extrapolate(der, steps):
            der, errors = dea3(der[0:-2], der[1:-1], der[2:], symmetric=False)
            return der, errors, steps[2:]

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

        def _get_best_estimate(der, errors, steps, shape):
            errors += _add_error_to_outliers(der)
            ix = _get_arg_min(errors)
            final_step = steps.flat[ix].reshape(shape)
            err = errors.flat[ix].reshape(shape)
            return der.flat[ix].reshape(shape), final_step, err

        results, steps, shape = _vstack(step_evaluates, self._steps)
        r_conv, _, _ = _vstack(step_evaluates_convoluted, self._steps)

        der1, errors1, steps = richardson(results, r_conv, steps)
        if len(der1) > 2:
            der1, errors1, steps = _wynn_extrapolate(der1, steps)
        der, final_step, err = _get_best_estimate(der1, errors1, steps, shape)
        return der


EPS = np.finfo(float).eps
_EPS = EPS
_TINY = np.finfo(float).tiny


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


class Richardson(object):

    """
    Extrapolates as sequence with Richardsons method

    Notes
    -----
    Suppose you have series expansion that goes like this

    L = f(h) + a0 * h^p_0 + a1 * h^p_1+ a2 * h^p_2 + ...

    where p_i = order + step * i  and f(h) -> L as h -> 0, but f(0) != L.

    If we evaluate the right hand side for different stepsizes h
    we can fit a polynomial to that sequence of approximations.
    This is exactly what this class does.

    Example
    -------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> n = 3
    >>> Ei = np.zeros((n,1))
    >>> h = np.zeros((n,1))
    >>> linfun = lambda i : np.linspace(0, np.pi/2., 2**(i+5)+1)
    >>> for k in np.arange(n):
    ...    x = linfun(k)
    ...    h[k] = x[1]
    ...    Ei[k] = np.trapz(np.sin(x),x)
    >>> En, err, step = nd.Richardson(step=1, order=1)(Ei, h)
    >>> truErr = Ei-1.
    >>> (truErr, err, En)
    (array([[ -2.00805680e-04],
           [ -5.01999079e-05],
           [ -1.25498825e-05]]), array([[ 0.00160242]]), array([[ 1.]]))

    """

    def __init__(self, step_ratio=2.0, step=1, order=1, num_terms=2):
        self.num_terms = num_terms
        self.order = order
        self.step = step
        self.step_ratio = step_ratio

    def _r_matrix(self, num_terms):
        step = self.step
        i, j = np.ogrid[0:num_terms + 1, 0:num_terms]
        r_mat = np.ones((num_terms + 1, num_terms + 1))
        r_mat[:, 1:] = (1.0 / self.step_ratio) ** (i * (step * j + self.order))
        return r_mat

    def rule(self, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.num_terms + 1
        num_terms = min(self.num_terms, sequence_length - 1)
        if num_terms > 0:
            r_mat = self._r_matrix(num_terms)
            return linalg.pinv(r_mat)[0]
        return np.ones((1,))

    @staticmethod
    def _estimate_error(new_sequence, old_sequence, steps, rule):
        m = new_sequence.shape[0]
        mo = old_sequence.shape[0]
        cov1 = np.sum(rule**2)  # 1 spare dof
        fact = np.maximum(12.7062047361747 * np.sqrt(cov1), EPS * 10.)
        if mo < 2:
            return (np.abs(new_sequence) * EPS + steps) * fact
        if m < 2:
            delta = np.diff(old_sequence, axis=0)
            tol = max_abs(old_sequence[:-1], old_sequence[1:]) * fact
            err = np.abs(delta)
            converged = err <= tol
            abserr = err[-m:] + np.where(converged[-m:], tol[-m:] * 10,
                                abs(new_sequence - old_sequence[-m:]) * fact)
            return abserr
#         if mo>2:
#             res, abserr = dea3(old_sequence[:-2], old_sequence[1:-1],
#                               old_sequence[2:] )
#             return abserr[-m:] * fact
        err = np.abs(np.diff(new_sequence, axis=0)) * fact
        tol = max_abs(new_sequence[1:], new_sequence[:-1]) * EPS * fact
        converged = err <= tol
        abserr = err + np.where(converged, tol * 10,
                                abs(new_sequence[:-1] -
                                    old_sequence[-m+1:]) * fact)
        return abserr

    def extrapolate(self, sequence, steps):
        return self.__call__(sequence, steps)

    def __call__(self, sequence, new_sequence, steps):
        ne = sequence.shape[0]
        rule = self.rule(ne)
        nr = rule.size - 1
        m = ne - nr
        mm = min(ne, m+1)
        abserr = self._estimate_error(new_sequence[:mm], sequence, steps, rule)
        return new_sequence[:m], abserr[:m], steps[:m]
