import multiprocessing
import numpy as np
import os
from mot.cl_routines.derivatives.extrapolation import Richardson, dea3
from mot.cl_routines.mapping.run_procedure import RunProcedure
from ...utils import KernelInputBuffer, SimpleNamedCLFunction, KernelInputLocalMemory
from ...cl_routines.base import CLRoutine


__author__ = 'Robbert Harms'
__date__ = '2017-10-16'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class StepGenerator(object):
    """A base class for step generators.

    These step generators should return multiple (diminishing) steps at which to evaluate the derivative. The estimates
    at these steps are then extrapolated to estimate the final derivatives.
    """

    def generate_steps(self):
        """Generates a sequence of steps of decreasing magnitude.

        This has to generate steps of logarithmic increase or decrease.

        Returns:
            ndarray: a two matrix with various step sizes (rows) for each parameter (columns).
        """
        raise NotImplementedError()

    def get_step_ratio(self):
        """Get the ratio at which the steps diminish.

        Returns:
            float: the ratio at which the steps diminish
        """
        raise NotImplementedError()


class SimpleStepGenerator(StepGenerator):

    def __init__(self, base_step, step_ratio=2., nmr_steps=15):
        """Generate multiple decreasing steps for derivative calculation.

        This generates steps according to the rule:

            steps = base_step * step_ratio**-i, i=0, 1,.., nmr_steps-1.

        For example, a base step of 2 with a step ratio of 2 and with 4 steps
        results in the step sizes: [2.0, 1.0, 0.5, 0.25]

        A heuristic for the base step is max(log(1 + abs(x)), 1) where x is the mean of each of the parameters.

        Args:
            base_step (ndarray): Defines the start step, i.e., maximum step per parameter.
            step_ratio (float): The ratio between sequential steps, needs to be larger > 1.
            nmr_steps (int): the number of steps to generate.
        """
        self.base_step = np.asanyarray(base_step)
        self.step_ratio = step_ratio
        self.nmr_steps = nmr_steps

    def generate_steps(self):
        return np.tile(self.base_step, (self.nmr_steps, 1)) * (self.step_ratio ** -np.arange(self.nmr_steps))[:, None]

    def get_step_ratio(self):
        return self.step_ratio


class ExtrapolateGradient(object):

    def __init__(self, steps, step_ratio, result_shape):
        self._steps = steps
        self._step_ratio = step_ratio
        self._result_shape = result_shape

    def __call__(self, step_evaluates):
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

        der1, errors1, steps = richardson(results, steps)
        if len(der1) > 2:
            der1, errors1, steps = _wynn_extrapolate(der1, steps)
        der, final_step, err = _get_best_estimate(der1, errors1, steps, shape)
        return der


class Hessian(CLRoutine):

    def calculate(self, cl_func, kernel_data, nmr_inst_per_problem, parameters, step_generator=None,
                  codec=None, double_precision=False, parameter_scalings=None):
        """Calculate and return the Hessian of the given function at the given parameters.

        This calculates the Hessian with the *central even* approach with Taylor expansion order 2.

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

        step_evaluates = np.zeros((parameters.shape[0], steps.shape[0], nmr_params, nmr_params), dtype=np_dtype)

        all_kernel_data = dict(kernel_data)
        all_kernel_data.update({
            'parameters': KernelInputBuffer(parameters),
            'local_reduction_lls': KernelInputLocalMemory(np.float64),
            'parameter_scalings': KernelInputBuffer(parameter_scalings, offset_str='0'),
            'steps': KernelInputBuffer(steps, offset_str='0'),
            'step_evaluates': KernelInputBuffer(step_evaluates)
        })

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._get_wrapped_function(cl_func, parameters, nmr_inst_per_problem, codec, steps),
                             all_kernel_data, parameters.shape[0], double_precision=double_precision,
                             use_local_reduction=True)

        step_evaluates = all_kernel_data['step_evaluates'].get_data()

        extrapolate = ExtrapolateGradient(steps, step_generator.get_step_ratio(), (1, steps.shape[1]))

        def data_iterator():
            for param_ind in range(parameters.shape[0]):
                yield step_evaluates[param_ind]

        # if os.name == 'nt':  # In Windows there is no fork.
        return np.array(list(map(extrapolate, data_iterator())))
        # else:
        #     try:
        #         p = multiprocessing.Pool()
        #         return_data = np.array(list(p.imap(extrapolate, data_iterator())))
        #         p.close()
        #         p.join()
        #         return return_data
        #     except OSError:
        #         return np.array(list(map(extrapolate, data_iterator())))

    def _get_wrapped_function(self, ll_function, parameters, nmr_inst_per_problem, codec, steps):
        nmr_params = parameters.shape[1]

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

                for(uint i = 0; i < ceil(''' + str(nmr_inst_per_problem) + ''' 
                                         / (mot_float_type)workgroup_size); i++){

                    observation_ind = i * workgroup_size + local_id;

                    if(observation_ind < ''' + str(nmr_inst_per_problem) + '''){
                        data->local_reduction_lls[local_id] += ''' + ll_function.get_cl_function_name() + '''(
                            data, x, observation_ind);
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                double ll = 0;
                for(uint i = 0; i < get_local_size(0); i++){
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
            
            void compute_step(mot_data_struct* data, mot_float_type* x_input, mot_float_type f_x_input, 
                              mot_float_type* parameter_steps, global mot_float_type* step_evaluate_ptr){
                
                mot_float_type perturbation[''' + str(nmr_params) + '''];
                mot_float_type calc;
                
                for(uint px = 0; px < ''' + str(nmr_params) + '''; px++){
                    calc = (
                          eval_step(data, x_input, 
                                    px, parameter_steps[px] / data->parameter_scalings[px],
                                    px, parameter_steps[px] / data->parameter_scalings[px])
                        + eval_step(data, x_input, 
                                    px, -parameter_steps[px] / data->parameter_scalings[px],
                                    px, -parameter_steps[px] / data->parameter_scalings[px])
                        - 2 * f_x_input
                        ) / (4 * parameter_steps[px] * parameter_steps[px]);
                    
                    if(get_local_id(0) == 0){
                        step_evaluate_ptr[px + px * ''' + str(nmr_params) + '''] = calc;
                    }
                    
                    for(uint py = px + 1; py < ''' + str(nmr_params) + '''; py++){
                        calc = (
                              eval_step(data, x_input, 
                                        px, parameter_steps[px] / data->parameter_scalings[px],
                                        py, parameter_steps[py] / data->parameter_scalings[py])
                            - eval_step(data, x_input, 
                                        px, parameter_steps[px] / data->parameter_scalings[px],
                                        py, -parameter_steps[py] / data->parameter_scalings[py])
                            - eval_step(data, x_input, 
                                        px, -parameter_steps[px] / data->parameter_scalings[px],
                                        py, parameter_steps[py] / data->parameter_scalings[py])
                            + eval_step(data, x_input, 
                                        px, -parameter_steps[px] / data->parameter_scalings[px],
                                        py, -parameter_steps[py] / data->parameter_scalings[py])
                        ) / (4 * parameter_steps[px] * parameter_steps[py]);

                        if(get_local_id(0) == 0){
                            step_evaluate_ptr[py + px * ''' + str(nmr_params) + '''] = calc;
                            step_evaluate_ptr[px + py * ''' + str(nmr_params) + '''] = calc;
                        }
                    }
                }    
            }
            
            void compute(mot_data_struct* data){
                uint param_ind;
                double eval;
                double f_x_input;

                mot_float_type x_input[''' + str(nmr_params) + '''];
                mot_float_type parameter_steps[''' + str(nmr_params) + '''];
                global mot_float_type* step_evaluate_ptr;
                
                for(param_ind = 0; param_ind < ''' + str(nmr_params) + '''; param_ind++){
                    x_input[param_ind] = data->parameters[param_ind];
                }
                f_x_input = calculate_function(data, x_input);

                for(uint step_ind = 0; step_ind < ''' + str(steps.shape[0]) + '''; step_ind++){ 
                    for(param_ind = 0; param_ind < ''' + str(nmr_params) + '''; param_ind++){
                        parameter_steps[param_ind] = data->steps[step_ind * ''' + str(nmr_params) + ''' + param_ind];
                    }
                    
                    step_evaluate_ptr = data->step_evaluates + step_ind * ''' + str(nmr_params**2) + ''';
                    compute_step(data, x_input, f_x_input, parameter_steps, step_evaluate_ptr);
                }
            }
        '''
        return SimpleNamedCLFunction(func, 'compute')


class Gradient(CLRoutine):

    def calculate(self, cl_func, kernel_data, nmr_inst_per_problem, parameters, step_generator=None,
                  codec=None, double_precision=False, parameter_scalings=None):
        """Calculate and return the gradient of the given function at the given parameters.

        This calculates the gradient using the central differencing method (``(f(x + h) - f(x - h)) / 2``)
        with Taylor expansion order 2.

        After calculating the derivative with various decreasing step sizes we use the Richardson extrapolation method
        to approximate the final derivatives based on an extrapolation of the results over the step sizes.

        This code is a CL accelerated version of the default gradient calculation of the NumDiffTool package
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

        if step_generator is None:
            step_generator = SimpleStepGenerator(np.ones(parameters.shape[1]))
        steps = step_generator.generate_steps().astype(np_dtype)

        if parameter_scalings is None:
            parameter_scalings = np.ones(parameters.shape[1], dtype=np.float64)
        parameter_scalings = np.asarray(parameter_scalings, dtype=np.float64)

        step_evaluates = np.zeros((parameters.shape[0], steps.shape[0], steps.shape[1]), dtype=np_dtype)

        all_kernel_data = dict(kernel_data)
        all_kernel_data.update({
            'parameters': KernelInputBuffer(parameters),
            'local_reduction_lls': KernelInputLocalMemory(np.float64),
            'parameter_scalings': KernelInputBuffer(parameter_scalings, offset_str='0'),
            'steps': KernelInputBuffer(steps, offset_str='0'),
            'step_evaluates': KernelInputBuffer(step_evaluates)
        })

        runner = RunProcedure(**self.get_cl_routine_kwargs())
        runner.run_procedure(self._get_wrapped_function(cl_func, parameters, nmr_inst_per_problem, codec, steps),
                             all_kernel_data, parameters.shape[0], double_precision=True, use_local_reduction=True)

        step_evaluates = all_kernel_data['step_evaluates'].get_data()

        extrapolate = ExtrapolateGradient(steps, step_generator.get_step_ratio(), (1, steps.shape[1]))

        def data_iterator():
            for param_ind in range(parameters.shape[0]):
                yield step_evaluates[param_ind]

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

                for(uint i = 0; i < ceil(''' + str(nmr_inst_per_problem) + ''' 
                                         / (mot_float_type)workgroup_size); i++){

                    observation_ind = i * workgroup_size + local_id;

                    if(observation_ind < ''' + str(nmr_inst_per_problem) + '''){
                        data->local_reduction_lls[local_id] += ''' + ll_function.get_cl_function_name() + '''(
                            data, x, observation_ind);
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                
                double ll = 0;
                for(uint i = 0; i < get_local_size(0); i++){
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
            
            double eval_parameter_steps(mot_data_struct* data, mot_float_type* parameters, mot_float_type* perturbation){
                mot_float_type x_tmp[''' + str(nmr_params) + '''];
                
                double plus_step;
                double min_step;
                
                // plus step        
                for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                    x_tmp[i] = (parameters[i] + perturbation[i] / data->parameter_scalings[i]);
                }
                apply_bounds(data, x_tmp);
                plus_step = calculate_function(data, x_tmp);
                
                
                // min step        
                for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                    x_tmp[i] = (parameters[i] - perturbation[i] / data->parameter_scalings[i]);
                }
                apply_bounds(data, x_tmp);
                min_step = calculate_function(data, x_tmp);
                
                return (plus_step - min_step) / 2.0;
            }

            void compute(mot_data_struct* data){
                uint param_ind;
                double eval;
                double f_x_input;
                
                mot_float_type x_input[''' + str(nmr_params) + '''];
                mot_float_type perturbation[''' + str(nmr_params) + '''] = {0};
                
                for(param_ind = 0; param_ind < ''' + str(nmr_params) + '''; param_ind++){
                    x_input[param_ind] = data->parameters[param_ind];
                }
                f_x_input = calculate_function(data, x_input);
                
                for(uint step_ind = 0; step_ind < ''' + str(steps.shape[0]) + '''; step_ind++){ 
                    for(param_ind = 0; param_ind < ''' + str(nmr_params) + '''; param_ind++){
                        perturbation[param_ind] = data->steps[step_ind * ''' + str(nmr_params) + ''' + param_ind];
                        
                        eval = eval_perturbation(data, x_input, perturbation);
                        
        '''
        # We divide the numerical derivative by the step size.
        # In between these two steps is also the place where higher order Taylor approximations should go.
        # see https://numdifftools.readthedocs.io/en/latest/src/numerical/derivest.html#richardson-extrapolation-
        # methodology-applied-to-derivative-estimation
        func += '''
                        eval /= perturbation[param_ind];
                        
                        if(get_local_id(0) == 0){
                            data->step_evaluates[step_ind * ''' + str(nmr_params) + ''' + param_ind] = eval;
                        }
                        
                        perturbation[param_ind] = 0;
                    }
                }
            }
        '''
        return SimpleNamedCLFunction(func, 'compute')
