import pyopencl as cl
import numpy as np
from .cl_environments import CLEnvironmentFactory
from .tools import get_cl_double_extension_definer, \
    get_read_write_cl_mem_flags, set_correct_cl_data_type, get_read_only_cl_mem_flags, get_write_only_cl_mem_flags, \
    ParameterCLCodeGenerator


__author__ = 'Robbert Harms'
__date__ = "2014-09-25"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CLToPythonCallbacks(object):

    def __init__(self, model, cl_environment=None):
        """Generate python callbacks for CL functions.

        This class is capable of generating callback functions that can be used to calculate the CL objective
        functions using a python callback.

        Args:
            model (AbstractModel): An instance of an AbstractModel, this contains the data and the
                functions we need to generate the callback functions.
            cl_environments (CLEnvironment, optional): The default cl environment to use for the callback CL bindings.
                If none given it defaults to use the first CPU device found. The CPU, since this means we don't have to
                send the memory to the GPU device.
        """
        super(CLToPythonCallbacks, self).__init__()
        if cl_environment:
            self._cl_environment = cl_environment
        else:
            self._cl_environment = CLEnvironmentFactory.single_device(cl_device_type=cl.device_type.CPU,
                                                                      compile_flags=('-cl-strict-aliasing',),
                                                                      fallback_to_any_device_type=True)[0]
        self._model = model
        self._var_data_dict = self._model.get_problems_var_data()
        self._prtcl_data_dict = self._model.get_problems_prtcl_data()
        self._fixed_data_dict = self._model.get_problems_fixed_data()
        self._clean_data_dicts()
        self._cl_environment_items_cache = {}

    def get_residual_cb(self, voxel_index, decode_params=False, cl_environment=None):
        """Get the residual generator callback. This calculates the residual of data minus estimation.

        Args:
            voxel_index (int): The index (in the list of ROI voxels) of the voxel we will return the evaluate CB for.
            decode_params (boolean): If we want to include decoding the parameters to model space.
                If true, the callback function assumes the parameters are in opt space.
            cl_environment (CLEnvironment): The environment to use for this function.
                If None, the global is taken from the constructor.

        Returns:
            number: The value for: func(params) = ydata - f(xdata, params)
        """
        if not cl_environment:
            cl_environment = self._cl_environment

        cl_param_decode_func = None
        if decode_params:
            if self._model.get_parameter_codec():
                cl_param_decode_func = self._model.get_parameter_codec().get_cl_decode_function('decodeParameters')

        var_data_dict = self._get_var_data_dict(voxel_index)
        kernel = self._get_function_eval_kernel(
            self._model.get_model_eval_function('evaluateModel'),
            self._model.get_observation_return_function('getObservation'),
            cl_param_decode_func,
            self._model.get_nmr_estimable_parameters(),
            var_data_dict,
            self._prtcl_data_dict,
            self._fixed_data_dict,
            cl_environment)

        data_buffers = []
        param_buf = cl.Buffer(cl_environment.context, cl.mem_flags.READ_ONLY,
                              size=np.dtype(np.float64).itemsize * self._model.get_nmr_estimable_parameters())
        data_buffers.append(param_buf)

        residuals = np.zeros((self._model.get_nmr_inst_per_problem(), ), dtype=np.float64, order='C')
        residuals_buf = cl.Buffer(cl_environment.context, get_write_only_cl_mem_flags(cl_environment),
                                  hostbuf=residuals)
        data_buffers.append(residuals_buf)

        data_buffers.extend(self._get_var_data_buffers(var_data_dict, cl_environment))
        data_buffers.extend(self._get_protocol_and_fixed_data_buffers(cl_environment))

        queue = self._get_queue(cl_environment)

        def eval_cb(x):
            cl.enqueue_copy(queue, param_buf, x.astype(np.float64, order='C'), is_blocking=True)
            kernel.evaluate(queue, (int(self._model.get_nmr_inst_per_problem()), ), None, *data_buffers)
            cl.enqueue_copy(queue, residuals, residuals_buf, is_blocking=True)
            return residuals.copy()

        return eval_cb

    def get_model_eval_cb(self, voxel_index, decode_params=False, cl_environment=None):
        """Get the model evaluate callback which returns for each measurement instance a estimated value.

        Args:
            voxel_index (int): The index (in the list of ROI voxels) of the voxel we will return the evaluate CB for.
            decode_params (boolean): If we want to include decoding the parameters to model space. If true,
                the callback function assumes the parameters are in opt space.
            cl_environment (CLEnvironment): The environment to use for this function.
                If None, the global is taken from the constructor.

        Returns:
            A callback function that has arguments 'x', an vector with the parameters we want to evaluate. The function
            returns a vector 'eval' that contains the model results for each measurement instance (protocol line).
        """
        if not cl_environment:
            cl_environment = self._cl_environment

        cl_param_decode_func = None
        if decode_params:
            if self._model.get_parameter_codec():
                cl_param_decode_func = self._model.get_parameter_codec().get_cl_decode_function('decodeParameters')

        var_data_dict = self._get_var_data_dict(voxel_index)
        kernel = self._get_function_eval_kernel(
            self._model.get_model_eval_function('evaluateModel'),
            None,
            cl_param_decode_func,
            self._model.get_nmr_estimable_parameters(),
            var_data_dict,
            self._prtcl_data_dict,
            self._fixed_data_dict,
            cl_environment)

        data_buffers = []
        param_buf = cl.Buffer(cl_environment.context, cl.mem_flags.READ_ONLY,
                              size=np.dtype(np.float64).itemsize * self._model.get_nmr_estimable_parameters())
        data_buffers.append(param_buf)

        evals = np.zeros((self._model.get_nmr_inst_per_problem(), ), dtype=np.float64, order='C')
        evals_buf = cl.Buffer(cl_environment.context, get_write_only_cl_mem_flags(cl_environment), hostbuf=evals)
        data_buffers.append(evals_buf)

        data_buffers.extend(self._get_var_data_buffers(var_data_dict, cl_environment))
        data_buffers.extend(self._get_protocol_and_fixed_data_buffers(cl_environment))

        queue = self._get_queue(cl_environment)

        def eval_cb(x):
            cl.enqueue_copy(queue, param_buf, x.astype(np.float64, order='C'), is_blocking=True)
            kernel.evaluate(queue, (int(self._model.get_nmr_inst_per_problem()), ), None, *data_buffers)
            cl.enqueue_copy(queue, evals, evals_buf, is_blocking=True)
            return evals.copy()

        return eval_cb

    def get_objective_cb(self, voxel_index, decode_params=False, cl_environment=None):
        """Get the model objective function calculation callback.

        This callback evaluates the model at the given position and applies the, in the model defined, noise model
        to end up with one value, the objective value.

        Args:
            voxel_index (int): The index (in the list of ROI voxels) of the voxel we will return the evaluate CB for.
            decode_params (boolean): If we want to include decoding the parameters to model space.
                If true, the callback function assumes the parameters are in opt space.
            cl_environment (CLEnvironment): The environment to use for this function.
                If None, the global is taken from the constructor.

        Returns:
            A callback function that has arguments 'x', an vector with the parameters we want to evaluate. The function
            returns a single double value that represents the objective function at the point of the given parameters.

            More precise the CB function looks like:
                func(params) = noise_model(ydata, f(xdata, params))
        """
        if not cl_environment:
            cl_environment = self._cl_environment

        cl_param_decode_func = None
        if decode_params:
            if self._model.get_parameter_codec():
                cl_param_decode_func = self._model.get_parameter_codec().get_cl_decode_function('decodeParameters')

        var_data_dict = self._get_var_data_dict(voxel_index)
        kernel = self._get_objective_kernel(
            self._model.get_objective_function('calculateObjective'),
            cl_param_decode_func,
            self._model.get_nmr_estimable_parameters(),
            var_data_dict,
            self._prtcl_data_dict,
            self._fixed_data_dict,
            cl_environment)

        data_buffers = []
        param_buf = cl.Buffer(cl_environment.context, cl.mem_flags.READ_ONLY,
                              size=np.dtype(np.float64).itemsize * self._model.get_nmr_estimable_parameters())
        data_buffers.append(param_buf)

        errors = np.zeros((1, ), dtype=np.float64, order='C')
        errors_buf = cl.Buffer(cl_environment.context, get_write_only_cl_mem_flags(cl_environment), hostbuf=errors)
        data_buffers.append(errors_buf)

        data_buffers.extend(self._get_var_data_buffers(var_data_dict, cl_environment))
        data_buffers.extend(self._get_protocol_and_fixed_data_buffers(cl_environment))

        queue = self._get_queue(cl_environment)

        def eval_cb(x):
            cl.enqueue_copy(queue, param_buf, x.astype(np.float64, order='C'), is_blocking=True)
            kernel.evaluate(queue, (1, ), None, *data_buffers)
            cl.enqueue_copy(queue, errors, errors_buf, is_blocking=True)
            return errors[0]

        return eval_cb

    def get_parameter_encode_cb(self, cl_environment=None):
        """Get the callback function for encoding parameters.

        Args:
            cl_environment (CLEnvironment): The cl environment to use,
                if none given we use the cl environment defined in the constructor.

        Returns:
            A python callback function that takes as parameter an numpy array of size (1, n) and that returns as output
             the encoded parameters.
        """
        if not cl_environment:
            cl_environment = self._cl_environment

        if cl_environment in self._cl_environment_items_cache:
            cl_items = self._cl_environment_items_cache[cl_environment]
            if cl_items.param_encode_cb:
                return cl_items.param_encode_cb
        else:
            self._cl_environment_items_cache.update({cl_environment: _CLEnvironmentsCachedItems()})

        codec = self._model.get_parameter_codec()
        kernel = self._get_space_transformer_kernel(codec.get_cl_encode_function('encodeParameters'),
                                                    'encodeParameters', codec.get_nmr_parameters(), cl_environment)
        queue = self._get_queue(cl_environment)
        read_write_flags = get_read_write_cl_mem_flags(cl_environment)

        def encode_cb(params_model_space):
            params = params_model_space.copy()
            param_buf = cl.Buffer(cl_environment.context, read_write_flags, hostbuf=params)
            kernel.transformParameterSpace(queue, (1, ), None, param_buf)
            cl.enqueue_copy(queue, params, param_buf, is_blocking=True)
            return params

        self._cl_environment_items_cache[cl_environment].param_encode_cb = encode_cb

        return encode_cb

    def get_parameter_decode_cb(self, cl_environment=None):
        """Get the callback function for decoding parameters.

        Args:
            cl_environment (CLEnvironment): The cl environment to use, if none given we
                use the cl environment defined in the constructor.

        Returns:
            A python callback function that takes as parameter an numpy array of size (1, n) and that returns as output
            the decoded parameters.
        """
        if not cl_environment:
            cl_environment = self._cl_environment

        if cl_environment in self._cl_environment_items_cache:
            cl_items = self._cl_environment_items_cache[cl_environment]
            if cl_items.param_decode_cb:
                return cl_items.param_decode_cb
        else:
            self._cl_environment_items_cache.update({cl_environment: _CLEnvironmentsCachedItems()})

        codec = self._model.get_parameter_codec()
        kernel = self._get_space_transformer_kernel(codec.get_cl_decode_function('decodeParameters'),
                                                    'decodeParameters', codec.get_nmr_parameters(), cl_environment)
        queue = self._get_queue(cl_environment)
        read_write_flags = get_read_write_cl_mem_flags(cl_environment)

        def decode_cb(params_enc_space):
            params = params_enc_space.copy()
            param_buf = cl.Buffer(cl_environment.context, read_write_flags, hostbuf=params)
            kernel.transformParameterSpace(queue, (1, ), None, param_buf)
            cl.enqueue_copy(queue, params, param_buf, is_blocking=True)
            return params

        self._cl_environment_items_cache[cl_environment].param_decode_cb = decode_cb
        return decode_cb

    def get_final_parameter_transform_cb(self, cl_environment=None):
        """Get the final parameter transformations callback function.

        Args:
            cl_environment (CLEnvironment): The cl environment to use, if none given we use the
                cl environment globally defined in the constructor.

        Returns:
            A python callback function that takes as parameter an numpy array of size (1, n) and that returns as output
            an array of the same shape but with the parameters decoded to their final state.
        """
        if not cl_environment:
            cl_environment = self._cl_environment

        if cl_environment in self._cl_environment_items_cache:
            cl_items = self._cl_environment_items_cache[cl_environment]
            if cl_items.final_param_transform_cb:
                return cl_items.final_param_transform_cb
        else:
            self._cl_environment_items_cache.update({cl_environment: _CLEnvironmentsCachedItems()})

        prtcl_and_fixed_data_buffers = self._get_protocol_and_fixed_data_buffers(cl_environment)

        cl_transform_func = self._model.get_final_parameter_transformations('applyFinalParameterTransformations')

        if cl_transform_func:
            kernel = self._get_final_param_transform_kernel(cl_transform_func,
                                                            self._model.get_nmr_estimable_parameters(),
                                                            self._prtcl_data_dict, self._fixed_data_dict,
                                                            cl_environment)
            queue = self._get_queue(cl_environment)
            read_write_flags = get_read_write_cl_mem_flags(cl_environment)

            def final_param_transform_cb(params):
                data_buffers = []
                x = params.copy()
                param_buf = cl.Buffer(cl_environment.context, read_write_flags, hostbuf=x)

                data_buffers.append(param_buf)
                data_buffers.extend(prtcl_and_fixed_data_buffers)
                kernel.transform(queue, (1, ), None, *data_buffers)
                cl.enqueue_copy(queue, x, param_buf, is_blocking=True)
                return x
        else:
            def final_param_transform_cb(params):
                return params

        self._cl_environment_items_cache[cl_environment].final_param_transform_cb = final_param_transform_cb
        return final_param_transform_cb

    def get_log_prior_cb(self, cl_environment=None):
        """Get the callback function for the log prior.

        Args:
            cl_environment (CLEnvironment): The cl environment to use, if none given we use the
                cl environment globally defined in the constructor.

        Returns:
            A python callback function that takes as parameter an numpy array of size (1, n) and that returns as output
            a single scalar value with the prior value for the given parameters.
        """
        if not cl_environment:
            cl_environment = self._cl_environment

        if cl_environment in self._cl_environment_items_cache:
            cl_items = self._cl_environment_items_cache[cl_environment]
            if cl_items.final_param_transform_cb:
                return cl_items.final_param_transform_cb
        else:
            self._cl_environment_items_cache.update({cl_environment: _CLEnvironmentsCachedItems()})

        queue = self._get_queue(cl_environment)
        read_only_flags = get_read_only_cl_mem_flags(cl_environment)
        write_only_flags = get_write_only_cl_mem_flags(cl_environment)

        result_buffer = cl.Buffer(cl_environment.context, write_only_flags, hostbuf=np.array((1,), dtype=np.float64))

        kernel = self._get_log_prior_kernel(self._model.get_log_prior_function('getLogPrior'),
                                            'getLogPrior', self._model.get_nmr_estimable_parameters(), cl_environment)

        def encode_cb(params_model_space):
            param_buf = cl.Buffer(cl_environment.context, read_only_flags, hostbuf=params_model_space)
            kernel.calculateLogPrior(queue, (1, ), None, param_buf, result_buffer)
            prior_result = np.array((1,), dtype=np.float64)
            cl.enqueue_copy(queue, prior_result, result_buffer, is_blocking=True)
            return prior_result[0]

        self._cl_environment_items_cache[cl_environment].param_encode_cb = encode_cb

        return encode_cb

    def _get_log_prior_kernel(self, cl_func, cl_func_name, nmr_params, cl_environment):
        """Get the kernel source code for the encode and decode operations."""
        kernel_source = get_cl_double_extension_definer(cl_environment.platform)
        kernel_source += cl_func
        kernel_source += '''
            __kernel void calculateLogPrior(constant double* x_global, global double* result){
                double x[''' + repr(nmr_params) + '''];
                for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                    x[i] = x_global[i];
                }
                result[0] = ''' + cl_func_name + '''(x);
            }
        '''
        return cl.Program(cl_environment.context, kernel_source).build(' '.join(cl_environment.compile_flags))

    def _get_space_transformer_kernel(self, cl_func, cl_func_name, nmr_params, environment):
        """Get the kernel source code for the encode and decode operations."""
        kernel_source = get_cl_double_extension_definer(environment.platform)
        kernel_source += cl_func
        kernel_source += '''
            __kernel void transformParameterSpace(global double* x_global){
                double x[''' + repr(nmr_params) + '''];

                for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                    x[i] = x_global[i];
                }

                ''' + cl_func_name + '''(x);

                for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                    x_global[i] = x[i];
                }
            }
        '''
        return cl.Program(environment.context, kernel_source).build(' '.join(environment.compile_flags))

    def _get_function_eval_kernel(self, cl_eval_func, cl_observation_func, cl_param_decode_func, nmr_params,
                                  var_data_dict, prtcl_data_dict, fixed_data_dict, environment):

        param_code_gen = ParameterCLCodeGenerator(environment.device, var_data_dict, prtcl_data_dict,
                                                  fixed_data_dict, add_var_data_multipliers=False)

        kernel_param_names = ['constant double* params', 'global double* evals']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = get_cl_double_extension_definer(environment.platform)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += cl_eval_func

        if cl_observation_func:
            kernel_source += cl_observation_func

        if cl_param_decode_func:
            kernel_source += cl_param_decode_func

        kernel_source += '''
            __kernel void evaluate(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);
                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''

                    double x[''' + repr(nmr_params) + '''];
                    for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        x[i] = params[i];
                    }
                    ''' + ('decodeParameters(x);' if cl_param_decode_func else '') + '''

                    evals[gid] = ''' + ('getObservation(&data, gid) - evaluateModel(&data, x, gid);'
                                        if cl_observation_func else 'evaluateModel(&data, x, gid);') + '''
                }
        '''
        return cl.Program(environment.context, kernel_source).build(' '.join(environment.compile_flags))

    def _get_objective_kernel(self, cl_objective_function, cl_param_decode_func, nmr_params,
                              var_data_dict, prtcl_data_dict, fixed_data_dict, environment):

        param_code_gen = ParameterCLCodeGenerator(environment.device, var_data_dict, prtcl_data_dict,
                                                  fixed_data_dict, add_var_data_multipliers=False)

        kernel_param_names = ['constant double* params', 'global double* fval']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = get_cl_double_extension_definer(environment.platform)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += cl_objective_function
        if cl_param_decode_func:
            kernel_source += cl_param_decode_func

        kernel_source += '''
            __kernel void evaluate(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''
                    double x[''' + repr(nmr_params) + '''];
                    for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        x[i] = params[i];
                    }
                    ''' + ('decodeParameters(x);' if cl_param_decode_func else '') + '''

                    fval[0] = calculateObjective(&data, x);
                }
        '''
        return cl.Program(environment.context, kernel_source).build(' '.join(environment.compile_flags))

    def _get_final_param_transform_kernel(self, transform_func, nmr_params, prtcl_data_dict, fixed_data_dict, environment):
        param_code_gen = ParameterCLCodeGenerator(environment.device, {}, prtcl_data_dict, fixed_data_dict,
                                                  add_var_data_multipliers=False)

        kernel_param_names = ['global double* params']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = get_cl_double_extension_definer(environment.platform)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += transform_func
        kernel_source += '''
            __kernel void transform(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''

                    double x[''' + repr(nmr_params) + '''];
                    for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        x[i] = params[i];
                    }

                    applyFinalParameterTransformations(&data, x);

                    for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        params[i] = x[i];
                    }
                }
        '''
        return cl.Program(environment.context, kernel_source).build(' '.join(environment.compile_flags))

    def _clean_data_dicts(self):
        """Set the data dicts to the right cl data type."""
        if self._var_data_dict:
            for key, data in self._var_data_dict.items():
                self._var_data_dict[key] = set_correct_cl_data_type(data)

        if self._prtcl_data_dict:
            for key, data in self._prtcl_data_dict.items():
                self._prtcl_data_dict[key] = set_correct_cl_data_type(data)

        if self._fixed_data_dict:
            for key, data in self._fixed_data_dict.items():
                self._fixed_data_dict[key] = set_correct_cl_data_type(data)

    def _get_protocol_and_fixed_data_buffers(self, cl_environment):
        read_only_flags = get_read_only_cl_mem_flags(cl_environment)

        if cl_environment in self._cl_environment_items_cache:
            cl_items = self._cl_environment_items_cache[cl_environment]
            if cl_items.prtcl_and_fixed_data_buffers:
                return cl_items.prtcl_and_fixed_data_buffers
        else:
            self._cl_environment_items_cache.update({cl_environment: _CLEnvironmentsCachedItems()})

        prtcl_and_fixed_data_buffers = []

        if self._prtcl_data_dict:
            for data in self._prtcl_data_dict.values():
                prtcl_and_fixed_data_buffers.append(cl.Buffer(cl_environment.context, read_only_flags, hostbuf=data))

        if self._fixed_data_dict:
            for data in self._fixed_data_dict.values():
                if isinstance(data, np.ndarray):
                    prtcl_and_fixed_data_buffers.append(cl.Buffer(cl_environment.context, read_only_flags,
                                                                  hostbuf=data))
                else:
                    prtcl_and_fixed_data_buffers.append(data)

        self._cl_environment_items_cache[cl_environment].prtcl_and_fixed_data_buffers = prtcl_and_fixed_data_buffers
        return prtcl_and_fixed_data_buffers

    def _get_var_data_dict(self, voxel_index):
        var_data_dict = {}
        for key, value in self._var_data_dict.items():
            if len(value.shape) > 1:
                var_data_dict[key] = value[voxel_index, :]
            else:
                var_data_dict[key] = value[voxel_index]
        return var_data_dict

    def _get_var_data_buffers(self, var_data_dict, cl_environment):
        var_data_buffers = []
        for data in var_data_dict.values():
            data = np.ascontiguousarray(data)
            var_data_buffers.append(cl.Buffer(cl_environment.context,
                                              get_read_only_cl_mem_flags(cl_environment), hostbuf=data))
        return var_data_buffers

    def _get_queue(self, cl_environment):
        if cl_environment in self._cl_environment_items_cache:
            cl_items = self._cl_environment_items_cache[cl_environment]
            if cl_items.queue:
                return cl_items.queue
        else:
            self._cl_environment_items_cache.update({cl_environment: _CLEnvironmentsCachedItems()})
        queue = cl_environment.get_new_queue()
        self._cl_environment_items_cache[cl_environment].queue = queue
        return queue


class _CLEnvironmentsCachedItems(object):

    def __init__(self):
        """A cache for items about CL environments.

        Instances of this class are meant to be used in a dictionary with as keys the cl_environment and as values
        these objects. It is basically a struct.

        Attributes:
            param_encode_cb (python callback): The python callback function for parameter encoding.
            param_decode_cb (python callback): The python callback function for parameter decoding.
            prtcl_and_fixed_data_buffers (list of pyopencl buffers): The list of prepared buffers for the protocol and
                fixed data buffers.
            final_param_transform_cb (python callback): The python callback function for final parameter transformations
            queue (pyopencl queue): The queue used.
        """
        self.param_encode_cb = None
        self.param_decode_cb = None
        self.prtcl_and_fixed_data_buffers = None
        self.final_param_transform_cb = None
        self.queue = None