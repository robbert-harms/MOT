import logging

import numpy as np
import pyopencl as cl
from mot.cl_functions import RanluxCL
from ...utils import results_to_dict, ParameterCLCodeGenerator, \
    get_float_type_def, initialize_ranlux
from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import Worker
from ...cl_routines.mapping.final_parameters_transformer import FinalParametersTransformer
from ...cl_routines.mapping.codec_runner import CodecRunner


__author__ = 'Robbert Harms'
__date__ = "2014-05-18"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


return_code_labels = {
    0: ['default', 'no return code specified'],
    1: ['found zero', 'sum of squares/evaluation below underflow limit'],
    2: ['converged', 'the relative error in the sum of squares/evaluation is at most tol'],
    3: ['converged', 'the relative error of the parameter vector is at most tol'],
    4: ['converged', 'both errors are at most tol'],
    5: ['trapped', 'by degeneracy; increasing epsilon might help'],
    6: ['exhausted', 'number of function calls exceeding preset patience'],
    7: ['failed', 'ftol<tol: cannot reduce sum of squares any further'],
    8: ['failed', 'xtol<tol: cannot improve approximate solution any further'],
    9: ['failed', 'gtol<tol: cannot improve approximate solution any further'],
    10: ['NaN', 'Function value is not-a-number or infinite']
}


class AbstractOptimizer(AbstractCLRoutine):

    def __init__(self, cl_environments, load_balancer, use_param_codec=True, patience=1,
                 optimizer_options=None, **kwargs):
        """Create a new optimizer that will minimize the given model with the given codec using the given environments.

        If the codec is None it is not used, if the environment is None, a suitable default environment should be
        created.

        Args:
            cl_environments (list of CLEnvironment): a list with the cl environments to use
            load_balancer (LoadBalancer): the load balance strategy to use
            use_param_codec (boolean): if this minimization should use the parameter codecs (param transformations)
            patience (int): The patience is used in the calculation of how many iterations to iterate the optimizer.
                The exact usage of this value of this parameter may change per optimizer.
            optimizer_options (dict): extra options one can set for the optimization routine. These are routine
                dependent.
        """
        self._use_param_codec = use_param_codec
        self.patience = patience or 1

        super(AbstractOptimizer, self).__init__(cl_environments, load_balancer, **kwargs)
        self._optimizer_options = optimizer_options or {}

    @property
    def use_param_codec(self):
        """Check if we will use the codec during optimization

        Returns:
            boolean: True if we will use the codec, false otherwise.
        """
        return self._use_param_codec

    @use_param_codec.setter
    def use_param_codec(self, use_param_codec):
        """Set if we will use the codec during optimization

        Args:
            use_param_codec (boolean): Set the value of use_param_codec.
        """
        self._use_param_codec = use_param_codec

    def minimize(self, model, init_params=None, full_output=False):
        """Minimize the given model with the given codec using the given environments.

        Args:
            model (AbstractModel): The model to minimize, instance of AbstractModel
            init_params (dict): A dictionary containing the results of a previous run, provides the starting point
            full_output (boolean): If set to true, the output is a list with the results and a dictionary
                with other outputs. If false, only return the results per problem instance.

        Returns:
            Either only the results per problem, or a list: (results, {}), depending on if full_output is set.
            Both the results dictionary as well as the extra output  dictionary should return results per
            problem instance.
        """


class AbstractParallelOptimizer(AbstractOptimizer):

    def __init__(self, cl_environments, load_balancer, use_param_codec=True, patience=1,
                 optimizer_options=None, **kwargs):
        super(AbstractParallelOptimizer, self).__init__(cl_environments, load_balancer, use_param_codec, patience,
                                                        optimizer_options=optimizer_options, **kwargs)
        self._logger = logging.getLogger(__name__)

    def minimize(self, model, init_params=None, full_output=False):
        self._logger.info('Entered optimization routine.')
        self._logger.info('We will use a {} precision float type for the calculations.'.format(
            'double' if model.double_precision else 'single'))
        for env in self.load_balancer.get_used_cl_environments(self.cl_environments):
            self._logger.info('Using device \'{}\' with compile flags {}'.format(str(env), str(env.compile_flags)))
        self._logger.info('The parameters we will optimize are: {0}'.format(model.get_optimized_param_names()))
        self._logger.info('We will use the optimizer {} '
                          'with patience {} and optimizer options {}'.format(self.get_pretty_name(),
                                                                             self.patience,
                                                                             self._optimizer_options))

        self._logger.info('Starting optimization preliminaries')
        starting_points = model.get_initial_parameters(init_params)
        nmr_params = starting_points.shape[1]

        var_data_dict = model.get_problems_var_data()
        protocol_data_dict = model.get_problems_protocol_data()
        model_data_dict = model.get_model_data()
        return_codes = np.zeros((starting_points.shape[0],), dtype=np.int32, order='C')

        space_transformer = CodecRunner(self.cl_environments, self.load_balancer, model.double_precision)
        param_codec = model.get_parameter_codec()
        if self.use_param_codec and param_codec:
            starting_points = space_transformer.encode(param_codec, starting_points)

        self._logger.info('Finished optimization preliminaries')
        self._logger.info('Starting optimization')

        workers = self._create_workers(self._get_worker_generator(self, model, starting_points, full_output,
                                                                  var_data_dict, protocol_data_dict, model_data_dict,
                                                                  nmr_params, return_codes, self._optimizer_options))
        self.load_balancer.process(workers, model.get_nmr_problems())

        self._logger.info('Finished optimization')

        self._logger.info('Calling finalize optimization results in the model')
        results = model.finalize_optimization_results(
            results_to_dict(starting_points, model.get_optimized_param_names()))

        self._logger.info('Optimization finished.')

        if full_output:
            return results, {'ReturnCodes': return_codes}
        return results

    def _get_worker_generator(self, *args):
        """Generate the worker generator callback for the function _create_workers()

        This is supposed to be overwritten by the implementing optimizer.

        Returns:
            the python callback for generating the worker
        """
        return lambda cl_environment: AbstractParallelOptimizerWorker(cl_environment, *args)


class AbstractParallelOptimizerWorker(Worker):

    def __init__(self, cl_environment, parent_optimizer, model, starting_points, full_output,
                 var_data_dict, protocol_data_dict, model_data_dict, nmr_params, return_codes, optimizer_options=None):
        super(AbstractParallelOptimizerWorker, self).__init__(cl_environment)

        self._optimizer_options = optimizer_options

        self._parent_optimizer = parent_optimizer

        self._model = model
        self._double_precision = model.double_precision
        self._nmr_params = nmr_params
        self._var_data_dict = var_data_dict
        self._protocol_data_dict = protocol_data_dict
        self._model_data_dict = model_data_dict
        self._return_codes = return_codes

        self._constant_buffers = self._generate_constant_buffers(self._protocol_data_dict, self._model_data_dict)

        param_codec = model.get_parameter_codec()
        self._use_param_codec = self._parent_optimizer.use_param_codec and param_codec

        self._starting_points = starting_points
        self._kernel = self._build_kernel()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start
        all_buffers, parameters_buffer, return_code_buffer = self._create_buffers(range_start, range_end)

        event = self._kernel.minimize(self._cl_run_context.queue, (nmr_problems, ), None, *all_buffers)

        event = cl.enqueue_map_buffer(self._cl_run_context.queue, parameters_buffer,
                                      cl.map_flags.READ, 0,
                                      [nmr_problems, self._starting_points.shape[1]],
                                      self._return_codes.dtype,
                                      order="C", wait_for=[event], is_blocking=False)[1]

        return cl.enqueue_map_buffer(self._cl_run_context.queue, return_code_buffer,
                                     cl.map_flags.READ, 0,
                                     [nmr_problems], self._return_codes.dtype,
                                     order="C", wait_for=[event], is_blocking=False)[1]

    def _create_buffers(self, range_start, range_end):
        nmr_problems = range_end - range_start

        all_buffers = []

        parameters_buffer = cl.Buffer(self._cl_run_context.context,
                                      cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                                      hostbuf=self._starting_points[range_start:range_end, :])
        all_buffers.append(parameters_buffer)

        return_code_buffer = cl.Buffer(self._cl_run_context.context,
                                       cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                       hostbuf=self._return_codes[range_start:range_end])
        all_buffers.append(return_code_buffer)

        for data in self._var_data_dict.values():
            all_buffers.append(cl.Buffer(self._cl_run_context.context,
                                         cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                         hostbuf=data.get_opencl_data()[range_start:range_end, ...]))

        all_buffers.extend(self._constant_buffers)

        if self._uses_random_numbers():
            all_buffers.append(initialize_ranlux(self._cl_environment, self._cl_run_context, nmr_problems))

        return all_buffers, parameters_buffer, return_code_buffer

    def _get_kernel_source(self):
        """Generate the kernel source for this optimization routine.

        By default this returns a full kernel source using information from _get_optimizer_cl_code()
        and _get_optimizer_call_name().

        One could overwrite this function to completely generate the kernel source, but most likely
        you would want to implement _get_optimizer_cl_code() and _get_optimizer_call_name().

        Args:
            data_state (OptimizeDataStateObject): The internal data state object
            cl_environment (CLEnvironment): The environment to create the kernel source for.

        Returns:
            str: The kernel source for this optimization routine.
        """
        cl_objective_function = self._model.get_objective_function('calculateObjective')
        cl_final_param_transform = self._model.get_final_parameter_transformations('applyFinalParamTransforms')

        nmr_params = self._nmr_params
        param_code_gen = ParameterCLCodeGenerator(self._cl_environment.device,
                                                  self._var_data_dict,
                                                  self._protocol_data_dict,
                                                  self._model_data_dict)

        kernel_param_names = ['global mot_float_type* params', 'global int* return_codes']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        if self._uses_random_numbers():
            kernel_param_names.append('global float4 *ranluxcltab')

        optimizer_call_args = 'x, (const void*) &data'

        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += str(param_code_gen.get_data_struct())

        if self._use_param_codec:
            param_codec = self._model.get_parameter_codec()
            decode_func = param_codec.get_cl_decode_function('decodeParameters')
            kernel_source += decode_func + "\n"

        if cl_final_param_transform:
            kernel_source += cl_final_param_transform

        kernel_source += cl_objective_function
        kernel_source += self._get_optimizer_cl_code()
        kernel_source += '''
            __kernel void minimize(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);
        '''

        if self._uses_random_numbers():
            kernel_source += '''
                    ranluxcl_state_t ranluxclstate;
                    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);
            '''
            optimizer_call_args += ', (void*) &ranluxclstate'

        kernel_source += '''
                    mot_float_type x[''' + str(nmr_params) + '''];
                    for(int i = 0; i < ''' + str(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                    }

                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''
                    return_codes[gid] = ''' + self._get_optimizer_call_name() + '''(''' + optimizer_call_args + ''');

                    ''' + ('decodeParameters(x);' if self._use_param_codec else '') + '''
                    ''' + ('applyFinalParamTransforms(&data, x);' if cl_final_param_transform else '') + '''

                    for(int i = 0; i < ''' + str(nmr_params) + '''; i++){
                        params[gid * ''' + str(nmr_params) + ''' + i] = x[i];
                    }
                }
        '''
        return kernel_source

    def _get_optimizer_cl_code(self):
        """Get the cl code for the implemented CL optimization routine.

        This is used by the default implementation of _get_kernel_source(). This function should return the
        CL code that is called during optimization for each voxel.

        By default this creates a CL function named 'evaluation' that can be called by the optimization routine.
        This default function takes into account the use of the parameter codec and calls the objective function
        of the model to actually evaluate the model.

        Args:
            data_state (OptimizeDataStateObject): The internal data state object

        Returns:
            str: The kernel source for the optimization routine.
        """
        optimizer_func = self._get_optimization_function()

        kernel_source = ''
        if self._use_param_codec:
            kernel_source += '''
                mot_float_type evaluate(mot_float_type* x, const void* data){
                    mot_float_type x_model[''' + str(self._nmr_params) + '''];
                    for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        x_model[i] = x[i];
                    }
                    decodeParameters(x_model);
                    return calculateObjective((optimize_data*)data, x_model);
                }
            '''
        else:
            kernel_source += '''
                mot_float_type evaluate(mot_float_type* x, const void* data){
                    return calculateObjective((optimize_data*)data, x);
                }
            '''

        if self._uses_random_numbers():
            rand_func = RanluxCL()
            kernel_source += '#define RANLUXCL_LUX 4' + "\n"
            kernel_source += rand_func.get_cl_header()
            kernel_source += rand_func.get_cl_code()

        kernel_source += optimizer_func.get_cl_header()
        kernel_source += optimizer_func.get_cl_code()
        return kernel_source

    def _get_optimization_function(self):
        """Return the optimization CLFunction object used by the implementing optimizer.

        This is a convenience function to avoid boilerplate in implementing _get_optimizer_cl_code().

        Returns:
            CLFunction: The optimization routine function that can provide the cl code for the actual
                optimization routine.
        """

    def _get_optimizer_call_name(self):
        """Get the call name of the optimization routine.

        This name is the name of the function called by the kernel to optimize a single voxel.

        Returns:
            str: The function name of the optimization function.
        """
        return ''

    def _uses_random_numbers(self):
        """Defines if the optimizer needs random numbers or not.

        This should be overwritten by the base class if it needs random numbers. If so, this class will
        take care of providing a rand() function.

        If this is set to True, this base class will provide a rand function and will pass an additional argument
        to the call of the optimizer 'void * rand_settings' containing the settings for the random number generator.

        Returns:
            boolean: if the optimizer needs a random number generator or not.
        """
        return False
