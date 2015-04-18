import warnings

import pyopencl as cl

from ...cl_environments import CLEnvironmentFactory
from pppe.cl_python_callbacks import CLToPythonCallbacks
from ...tools import get_read_only_cl_mem_flags, get_read_write_cl_mem_flags, \
    set_correct_cl_data_type, results_to_dict, ParameterCLCodeGenerator, get_cl_double_extension_definer
from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import PreferGPU, WorkerConstructor
from ...cl_routines.mapping.final_parameters_transformer import FinalParametersTransformer
from ...cl_routines.mapping.codec_runner import CodecRunner


__author__ = 'Robbert Harms'
__date__ = "2014-05-18"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractOptimizer(AbstractCLRoutine):

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=True, patience=1):
        """Create a new optimizer that will minimize the given model with the given codec using the given environments.

        If the codec is None it is not used, if the environment is None, a suitable default environment should be
        created.

        Args:
            cl_environments (list of CLEnvironment): a list with the cl environments to use
            load_balancer (LoadBalancer): the load balance strategy to use
            use_param_codec (boolean): if this minimization should use the parameter codecs (param transformations)
            patience (int): The patience is used in the calculation of how many iterations to iterate the optimizer.
                The exact semantical value of this parameter may change per optimizer.
        """
        self._use_param_codec = use_param_codec
        self.patience = patience

        if not load_balancer:
            load_balancer = PreferGPU()

        if not cl_environments:
            cl_environments = CLEnvironmentFactory.all_devices(compile_flags=('-cl-strict-aliasing',
                                                                              '-cl-no-signed-zeros'))

        if not patience:
            self.patience = 1

        super(AbstractOptimizer, self).__init__(cl_environments, load_balancer)

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
            Either only the results per problem, or a list: (results, {}) where the dictionary contains all
            other parameters (named) that can be returned.
        """


class AbstractParallelOptimizer(AbstractOptimizer):

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=True, patience=1):
        super(AbstractParallelOptimizer, self).__init__(cl_environments, load_balancer, use_param_codec, patience)
        self._automatic_apply_codec = True

    def minimize(self, model, init_params=None, full_output=False):
        starting_points = model.get_initial_parameters(init_params)
        data_state = self._get_data_state_object(model, starting_points)
        cl_environments = self.load_balancer.get_used_cl_environments(self.cl_environments)
        space_transformer = CodecRunner()

        param_codec = model.get_parameter_codec()
        if self.use_param_codec and param_codec and self._automatic_apply_codec:
            starting_points = space_transformer.encode(param_codec, starting_points)

        def minimizer_generator(cl_environment, start, end, buffered_dicts):
            warnings.simplefilter("ignore")

            prtcl_dbuf = buffered_dicts[0]
            fixed_dbuf = buffered_dicts[1]

            kernel_source = self._get_kernel_source(data_state, cl_environment)
            kernel = cl.Program(cl_environment.context, kernel_source).build(' '.join(cl_environment.compile_flags))

            return self._run_minimizer(starting_points, prtcl_dbuf, data_state.var_data_dict, fixed_dbuf,
                                       start, end, cl_environment, kernel)

        worker_constructor = WorkerConstructor()
        workers = worker_constructor.generate_workers(cl_environments, minimizer_generator,
                                                      data_dicts_to_buffer=(data_state.prtcl_data_dict,
                                                                            data_state.fixed_data_dict))

        self.load_balancer.process(workers, model.get_nmr_problems())

        optimized = FinalParametersTransformer(cl_environments=cl_environments, load_balancer=self.load_balancer).\
            transform(model, starting_points)
        results = model.post_optimization(results_to_dict(optimized, model.get_optimized_param_names()))
        if full_output:
            return results, {}
        return results

    def _get_data_state_object(self, model, init_params):
        return OptimizeDataStateObject(model, init_params.shape[1])

    def _get_kernel_source(self, data_state, cl_environment):
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
        cl_objective_function = data_state.model.get_objective_function('calculateObjective')
        nmr_params = data_state.nmr_params
        param_codec = data_state.model.get_parameter_codec()
        param_code_gen = ParameterCLCodeGenerator(cl_environment.device,
                                                  data_state.var_data_dict,
                                                  data_state.prtcl_data_dict,
                                                  data_state.fixed_data_dict)

        kernel_param_names = ['global double* params']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        use_param_codec = self.use_param_codec and param_codec and self._automatic_apply_codec

        kernel_source = ''
        kernel_source += get_cl_double_extension_definer(cl_environment.platform)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += cl_objective_function
        kernel_source += self._get_optimizer_cl_code(data_state)
        kernel_source += '''
            __kernel void minimize(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);

                    double x[''' + repr(nmr_params) + '''];
                    for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + repr(nmr_params) + ''' + i];
                    }

                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''
                    ''' + self._get_optimizer_call_name() + '''(x, (const void*) &data);
                    ''' + ('decodeParameters(x);' if use_param_codec else '') + '''

                    for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        params[gid * ''' + repr(nmr_params) + ''' + i] = x[i];
                    }
            }
        '''
        return kernel_source

    def _get_optimizer_cl_code(self, data_state):
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
        param_codec = data_state.model.get_parameter_codec()
        nmr_params = data_state.nmr_params
        use_param_codec = self.use_param_codec and param_codec and self._automatic_apply_codec

        optimizer_func = self._get_optimization_function(data_state)

        kernel_source = ''
        if use_param_codec:
            decode_func = param_codec.get_cl_decode_function('decodeParameters')
            kernel_source += decode_func + "\n"
            kernel_source += '''
                double evaluate(double* x, const void* data){
                    double x_model[''' + repr(nmr_params) + '''];
                    for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        x_model[i] = x[i];
                    }
                    decodeParameters(x_model);

                    return calculateObjective((optimize_data*)data, x_model);
                }
            '''
        else:
            kernel_source += '''
                double evaluate(double* x, const void* data){
                    return calculateObjective((optimize_data*)data, x);
                }
            '''
        kernel_source += optimizer_func.get_cl_header()
        kernel_source += optimizer_func.get_cl_code()
        return kernel_source

    def _get_optimization_function(self, data_state):
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

    def _run_minimizer(self, parameters, prtcl_data_buffers, var_data_dict, fixed_data_buffers,
                                start, end, environment, kernel):
        start = int(start)
        end = int(end)

        queue = environment.get_new_queue()
        nmr_problems = end - start

        read_only_flags = get_read_only_cl_mem_flags(environment)
        read_write_flags = get_read_write_cl_mem_flags(environment)

        data_buffers = []
        parameters_buf = cl.Buffer(environment.context, read_write_flags, hostbuf=parameters[start:end, :])
        data_buffers.append(parameters_buf)

        for data in var_data_dict.values():
            if len(data.shape) < 2:
                data_buffers.append(cl.Buffer(environment.context, read_only_flags, hostbuf=data[start:end]))
            else:
                data_buffers.append(cl.Buffer(environment.context, read_only_flags, hostbuf=data[start:end, :]))

        data_buffers.extend(prtcl_data_buffers)
        data_buffers.extend(fixed_data_buffers)

        local_range = None
        global_range = (nmr_problems, )
        kernel.minimize(queue, global_range, local_range, *data_buffers)

        event = cl.enqueue_copy(queue, parameters[start:end, :], parameters_buf, is_blocking=False)

        return queue, event


class AbstractSerialOptimizer(AbstractOptimizer):

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=True, patience=None):
        """The base class for serial optimization.

        Serial optimization is the process in which each voxel is optimized one at a time, regularly by a python
        optimization routine. For this to work all CL function in the model class need to be wrapped in a python
        callback function. This is taken care of automatically.
        """
        super(AbstractSerialOptimizer, self).__init__(cl_environments, load_balancer=load_balancer,
                                                      use_param_codec=use_param_codec)

    def minimize(self, model, init_params=None, full_output=False):
        space_transformer = CodecRunner()
        cl_environments = self.load_balancer.get_used_cl_environments(self.cl_environments)
        starting_points = model.get_initial_parameters(init_params)

        param_codec = model.get_parameter_codec()
        if self.use_param_codec and param_codec:
            starting_points = space_transformer.encode(param_codec, starting_points)

        optimized = self._minimize(model, starting_points, cl_environments[0])

        if self.use_param_codec and param_codec:
            optimized = space_transformer.decode(param_codec, optimized)

        optimized = FinalParametersTransformer(cl_environments=cl_environments, load_balancer=self.load_balancer).\
            transform(model, optimized)

        results = model.post_optimization(results_to_dict(optimized, model.get_optimized_param_names()))
        if full_output:
            return results, {}
        return results

    def _minimize(self, model, starting_points, cl_environment):
        cb_generator = CLToPythonCallbacks(model, cl_environment=cl_environment)

        for voxel_index in range(model.get_nmr_problems()):
            objective_cb = cb_generator.get_objective_cb(voxel_index, decode_params=True)
            x0 = starting_points[voxel_index, :]
            x_opt = self._minimize_single_voxel(objective_cb, x0)
            starting_points[voxel_index, :] = x_opt

        return starting_points

    def _minimize_single_voxel(self, objective_cb, x0):
        """Minimize a single voxel and return the results"""


class OptimizeDataStateObject(object):

    def __init__(self, model, nmr_params):
        self.model = model
        self.nmr_params = nmr_params
        self.var_data_dict = set_correct_cl_data_type(model.get_problems_var_data())
        self.prtcl_data_dict = set_correct_cl_data_type(model.get_problems_prtcl_data())
        self.fixed_data_dict = set_correct_cl_data_type(model.get_problems_fixed_data())
