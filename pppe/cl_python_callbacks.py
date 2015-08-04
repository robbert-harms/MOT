import warnings
import pyopencl as cl
import numpy as np
from .cl_environments import CLEnvironmentFactory
from pppe.cl_functions import RanluxCL
from .utils import get_cl_double_extension_definer, set_correct_cl_data_type, ParameterCLCodeGenerator, \
    initialize_ranlux


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
        self._state = _GeneratorState(model, cl_environment)

    def get_residual_cb(self, voxel_index, decode_params=False, cl_environment=None):
        """Get the residual generator callback. This calculates the residual of data minus estimation.

        Args:
            voxel_index (int): The index (in the list of ROI voxels) of the voxel we will return the evaluate CB for.
            decode_params (boolean): If we want to include decoding the parameters to model space.
                If true, the callback function assumes the parameters are in opt space.
            cl_environment (CLEnvironment): The environment to use for this function.
                If None, the global is taken from the constructor.

        Returns:
            python function: Which when calls returns the value for: func(params) = ydata - f(xdata, params)
        """
        generator = _ResidualCBGenerator(self._state)
        return generator.get_cb(voxel_index, decode_params, cl_environment)

    def get_model_eval_cb(self, voxel_index, decode_params=False, cl_environment=None):
        """Get the model evaluate callback which returns for each measurement instance an estimated value.

        Args:
            voxel_index (int): The index (in the list of ROI voxels) of the voxel we will return the evaluate CB for.
            decode_params (boolean): If we want to include decoding the parameters to model space. If true,
                the callback function assumes the parameters are in opt space.
            cl_environment (CLEnvironment): The environment to use for this function.
                If None, the global is taken from the constructor.

        Returns:
            python function: A callback function that has arguments 'x', an vector with the parameters we want
            to evaluate. The function returns a vector 'eval' that contains the model results for
            each measurement instance.
        """
        generator = _EvalCBGenerator(self._state)
        return generator.get_cb(voxel_index, decode_params, cl_environment)

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
        generator = _ObjectiveCBGenerator(self._state)
        return generator.get_cb(voxel_index, decode_params, cl_environment)

    def get_parameter_encode_cb(self, cl_environment=None):
        """Get the callback function for encoding parameters.

        Args:
            cl_environment (CLEnvironment): The cl environment to use,
                if none given we use the cl environment defined in the constructor.

        Returns:
            A python callback function that takes as parameter an numpy array of size (1, n) and that returns as output
             the encoded parameters.
        """
        generator = _CodecCBGenerator(self._state)
        return generator.get_encode_cb(cl_environment)

    def get_parameter_decode_cb(self, cl_environment=None):
        """Get the callback function for decoding parameters.

        Args:
            cl_environment (CLEnvironment): The cl environment to use, if none given we
                use the cl environment defined in the constructor.

        Returns:
            A python callback function that takes as parameter an numpy array of size (1, n) and that returns as output
            the decoded parameters.
        """
        generator = _CodecCBGenerator(self._state)
        return generator.get_decode_cb(cl_environment)

    def get_final_parameter_transform_cb(self, cl_environment=None):
        """Get the final parameter transformations callback function.

        Args:
            cl_environment (CLEnvironment): The cl environment to use, if none given we use the
                cl environment globally defined in the constructor.

        Returns:
            A python callback function that takes as parameter an numpy array of size (1, n) and that returns as output
            an array of the same shape but with the parameters decoded to their final state.
        """
        generator = _FinalTransformationCBGenerator(self._state)
        return generator.get_cb(cl_environment)

    def get_log_prior_cb(self, cl_environment=None):
        """Get the callback function for the log prior.

        Args:
            cl_environment (CLEnvironment): The cl environment to use, if none given we use the
                cl environment globally defined in the constructor.

        Returns:
            A python callback function that takes as parameter an numpy array of size (1, n) and that returns as output
            a single scalar value with the prior value for the given parameters.
        """
        generator = _LogPriorCBGenerator(self._state)
        return generator.get_cb(cl_environment)

    def get_proposal_cb(self, cl_environment=None):
        """Get the callback function for the proposal function used in sampling.

        Args:
            cl_environment (CLEnvironment): The cl environment to use, if none given we use the
                cl environment globally defined in the constructor.

        Returns:
            A python callback function that takes as parameters:
                - the parameter index in the list of parameters for the current parameter we want the proposal of
                - the current value of that parameter.
        """
        generator = _ProposalCBGenerator(self._state)
        return generator.get_cb(cl_environment)


class _GeneratorState(object):

    def __init__(self, model, cl_environment):
        """The CLToPythonCallbacks uses this class to keep track of the state.

        Since in most cases multiple callbacks are generated we can save the time of setting up the CL parts by
        keeping track of the state.

        This class uses the _CLEnvironmentsCachedItems to keep track of created CL buffers for protocol and fixed data
        buffers.

        Attributes:
            model: The model we are creating callbacks for
            var_data_dict: The variable data of the model
            prtcl_data_dict: The protocol data dict of the model
            fixed_data_dict: The fixed data dict of the model
            cl_environment_items_cache (_CLEnvironmentsCachedItems): The items cache for the buffers.
        """
        if cl_environment:
            self.cl_environment = cl_environment
        else:
            self.cl_environment = CLEnvironmentFactory.single_device(cl_device_type=cl.device_type.CPU,
                                                                      compile_flags=('-cl-strict-aliasing',),
                                                                      fallback_to_any_device_type=True)[0]
        self.model = model
        self.var_data_dict = self.model.get_problems_var_data()
        self.prtcl_data_dict = self.model.get_problems_prtcl_data()
        self.fixed_data_dict = self.model.get_problems_fixed_data()
        self.cl_environment_items_cache = {}
        self._clean_data_dicts()

    def _clean_data_dicts(self):
        """Set the data dicts to the right cl data type."""
        def set_correct(d):
            for key, data in d.items():
                d[key] = set_correct_cl_data_type(data)
        map(set_correct, [self.var_data_dict, self.prtcl_data_dict, self.fixed_data_dict])


class _CLEnvironmentsCachedItems(object):

    def __init__(self):
        """A cache for items about CL environments.

        Instances of this class are meant to be used in a dictionary with as keys the cl_environment and as values
        these objects.

        Attributes:
            param_encode_cb (python function): The python callback function for parameter encoding.
            param_decode_cb (python function): The python callback function for parameter decoding.
            prtcl_and_fixed_data_buffers (list of pyopencl buffers): The list of prepared buffers for the protocol and
                fixed data buffers.
            final_param_transform_cb (python function): The python callback function for final parameter transformations
            log_prior_cb (python function): for generating the prior
            queue (pyopencl queue): The queue used.
        """
        self.param_encode_cb = None
        self.param_decode_cb = None
        self.prtcl_and_fixed_data_buffers = None
        self.final_param_transform_cb = None
        self.log_prior_cb = None
        self.queue = None


class _BaseCBGenerator(object):

    def __init__(self, generator_state):
        self._state = generator_state

    def _get_var_data_dict(self, voxel_index):
        """Get all the variable data dictionary items for the indicated voxel.

        Args:
            voxel_index (int): The voxel we are interested in (linear index).

        Returns:
            dict: All the variable data dict items for that voxel.
        """
        var_data_dict = {}
        for key, value in self._state.var_data_dict.items():
            if len(value.shape) > 1:
                var_data_dict[key] = value[voxel_index, :]
            else:
                var_data_dict[key] = value[voxel_index]
        return var_data_dict

    def _get_queue(self, cl_environment):
        """Get a CL queue for the given CL environment.

        Args:
            cl_environment (CLEnvironment): The CL environment to get a queue for.

        Returns:
            A new CL queue, or a reference to a cached one.
        """
        if cl_environment in self._state.cl_environment_items_cache:
            cl_items = self._state.cl_environment_items_cache[cl_environment]
            if cl_items.queue:
                return cl_items.queue
        else:
            self._state.cl_environment_items_cache.update({cl_environment: _CLEnvironmentsCachedItems()})
        queue = cl_environment.get_new_queue()
        self._state.cl_environment_items_cache[cl_environment].queue = queue
        return queue

    def _get_protocol_and_fixed_data_buffers(self, cl_environment):
        read_only_flags = cl_environment.get_read_only_cl_mem_flags()

        if cl_environment in self._state.cl_environment_items_cache:
            cl_items = self._state.cl_environment_items_cache[cl_environment]
            if cl_items.prtcl_and_fixed_data_buffers:
                return cl_items.prtcl_and_fixed_data_buffers
        else:
            self._state.cl_environment_items_cache.update({cl_environment: _CLEnvironmentsCachedItems()})

        prtcl_and_fixed_data_buffers = []

        if self._state.prtcl_data_dict:
            for data in self._state.prtcl_data_dict.values():
                prtcl_and_fixed_data_buffers.append(cl.Buffer(cl_environment.context, read_only_flags, hostbuf=data))

        if self._state.fixed_data_dict:
            for data in self._state.fixed_data_dict.values():
                if isinstance(data, np.ndarray):
                    prtcl_and_fixed_data_buffers.append(cl.Buffer(cl_environment.context, read_only_flags,
                                                                  hostbuf=data))
                else:
                    prtcl_and_fixed_data_buffers.append(data)

        self._state.cl_environment_items_cache[cl_environment].prtcl_and_fixed_data_buffers = \
            prtcl_and_fixed_data_buffers
        return prtcl_and_fixed_data_buffers

    def _create_buffer(self, data_dict, cl_environment):
        """Create a buffer for the given data items.

        Args:
            data_dict (dict): The dictionary with the data items.
            cl_environment (CLEnvironment): The CL environment to get use.

        Returns:
            list: A list of buffers in linear order for each data in the dict.
        """
        data_buffers = []
        for data in data_dict.values():
            data = np.ascontiguousarray(data)
            data_buffers.append(cl.Buffer(cl_environment.context,
                                          cl_environment.get_read_only_cl_mem_flags(), hostbuf=data))
        return data_buffers


class _ResidualCBGenerator(_BaseCBGenerator):

    def get_cb(self, voxel_index, decode_params, cl_environment):
        """Get the residual generator callback. This calculates the residual of data minus estimation.

        Args:
            voxel_index (int): The index (in the list of ROI voxels) of the voxel we will return the evaluate CB for.
            decode_params (boolean): If we want to include decoding the parameters to model space.
                If true, the callback function assumes the parameters are in opt space.
            cl_environment (CLEnvironment): The environment to use for this function.
                If None, the global is taken from the constructor.

        Returns:
            python function: Which when calls returns the value for: func(params) = ydata - f(xdata, params)
        """
        if not cl_environment:
            cl_environment = self._state.cl_environment

        var_data_dict = self._get_var_data_dict(voxel_index)
        kernel = self.get_kernel(decode_params, var_data_dict, cl_environment)

        data_buffers = []
        param_buf = cl.Buffer(cl_environment.context, cl.mem_flags.READ_ONLY,
                              size=np.dtype(np.float64).itemsize * self._state.model.get_nmr_estimable_parameters())
        data_buffers.append(param_buf)

        residuals = np.zeros((self._state.model.get_nmr_inst_per_problem(), ), dtype=np.float64, order='C')
        residuals_buf = cl.Buffer(cl_environment.context, cl_environment.get_write_only_cl_mem_flags(),
                                  hostbuf=residuals)
        data_buffers.append(residuals_buf)

        data_buffers.extend(self._create_buffer(var_data_dict, cl_environment))
        data_buffers.extend(self._get_protocol_and_fixed_data_buffers(cl_environment))

        queue = self._get_queue(cl_environment)

        def eval_cb(x):
            cl.enqueue_copy(queue, param_buf, x.astype(np.float64, order='C'), is_blocking=True)
            kernel.evaluate(queue, (int(self._state.model.get_nmr_inst_per_problem()), ), None, *data_buffers)
            cl.enqueue_copy(queue, residuals, residuals_buf, is_blocking=True)
            return residuals.copy()

        return eval_cb

    def get_kernel(self, decode_params, var_data_dict, environment):
        nmr_params = self._state.model.get_nmr_estimable_parameters()

        prtcl_data_dict = self._state.prtcl_data_dict
        fixed_data_dict = self._state.fixed_data_dict

        param_code_gen = ParameterCLCodeGenerator(environment.device, var_data_dict, prtcl_data_dict,
                                                  fixed_data_dict, add_var_data_multipliers=False)

        kernel_param_names = ['constant double* params', 'global double* evals']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = get_cl_double_extension_definer(environment.platform)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += self._state.model.get_model_eval_function('evaluateModel')
        kernel_source += self._state.model.get_observation_return_function('getObservation')

        use_codec = False
        if decode_params:
            if self._state.model.get_parameter_codec():
                use_codec = True
                kernel_source += self._state.model.get_parameter_codec().get_cl_decode_function('decodeParameters')

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
                    ''' + ('decodeParameters(x);' if use_codec else '') + '''

                    evals[gid] = getObservation(&data, gid) - evaluateModel(&data, x, gid);
                }
        '''
        warnings.simplefilter("ignore")
        return cl.Program(environment.context, kernel_source).build(' '.join(environment.compile_flags))


class _EvalCBGenerator(_BaseCBGenerator):

    def get_cb(self, voxel_index, decode_params, cl_environment):
        """Get the model evaluate callback which returns for each measurement instance an estimated value.

        Args:
            voxel_index (int): The index (in the list of ROI voxels) of the voxel we will return the evaluate CB for.
            decode_params (boolean): If we want to include decoding the parameters to model space. If true,
                the callback function assumes the parameters are in opt space.
            cl_environment (CLEnvironment): The environment to use for this function.
                If None, the global is taken from the constructor.

        Returns:
            python function: A callback function that has arguments 'x', an vector with the parameters we want
            to evaluate. The function returns a vector 'eval' that contains the model results for
            each measurement instance.
        """
        if not cl_environment:
            cl_environment = self._state.cl_environment

        var_data_dict = self._get_var_data_dict(voxel_index)
        kernel = self._get_kernel(decode_params, var_data_dict, cl_environment)

        data_buffers = []
        param_buf = cl.Buffer(cl_environment.context, cl.mem_flags.READ_ONLY,
                              size=np.dtype(np.float64).itemsize * self._state.model.get_nmr_estimable_parameters())
        data_buffers.append(param_buf)

        evals = np.zeros((self._state.model.get_nmr_inst_per_problem(), ), dtype=np.float64, order='C')
        evals_buf = cl.Buffer(cl_environment.context, cl_environment.get_write_only_cl_mem_flags(), hostbuf=evals)
        data_buffers.append(evals_buf)

        data_buffers.extend(self._create_buffer(var_data_dict, cl_environment))
        data_buffers.extend(self._get_protocol_and_fixed_data_buffers(cl_environment))

        queue = self._get_queue(cl_environment)

        def eval_cb(x):
            cl.enqueue_copy(queue, param_buf, x.astype(np.float64, order='C'), is_blocking=True)
            kernel.evaluate(queue, (int(self._state.model.get_nmr_inst_per_problem()), ), None, *data_buffers)
            cl.enqueue_copy(queue, evals, evals_buf, is_blocking=True)
            return evals.copy()

        return eval_cb

    def _get_kernel(self, decode_params, var_data_dict, environment):
        nmr_params = self._state.model.get_nmr_estimable_parameters()

        prtcl_data_dict = self._state.prtcl_data_dict
        fixed_data_dict = self._state.fixed_data_dict

        param_code_gen = ParameterCLCodeGenerator(environment.device, var_data_dict, prtcl_data_dict,
                                                  fixed_data_dict, add_var_data_multipliers=False)

        kernel_param_names = ['constant double* params', 'global double* evals']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = get_cl_double_extension_definer(environment.platform)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += self._state.model.get_model_eval_function('evaluateModel')

        use_codec = False
        if decode_params:
            if self._state.model.get_parameter_codec():
                use_codec = True
                kernel_source += self._state.model.get_parameter_codec().get_cl_decode_function('decodeParameters')

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
                    ''' + ('decodeParameters(x);' if use_codec else '') + '''

                    evals[gid] = evaluateModel(&data, x, gid);
                }
        '''
        warnings.simplefilter("ignore")
        return cl.Program(environment.context, kernel_source).build(' '.join(environment.compile_flags))


class _ObjectiveCBGenerator(_BaseCBGenerator):

    def get_cb(self, voxel_index, decode_params, cl_environment):
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
            cl_environment = self._state.cl_environment

        var_data_dict = self._get_var_data_dict(voxel_index)
        kernel = self._get_kernel(decode_params, var_data_dict, cl_environment)

        data_buffers = []
        param_buf = cl.Buffer(cl_environment.context, cl.mem_flags.READ_ONLY,
                              size=np.dtype(np.float64).itemsize * self._state.model.get_nmr_estimable_parameters())
        data_buffers.append(param_buf)

        errors = np.zeros((1, ), dtype=np.float64, order='C')
        errors_buf = cl.Buffer(cl_environment.context, cl_environment.get_write_only_cl_mem_flags(), hostbuf=errors)
        data_buffers.append(errors_buf)

        data_buffers.extend(self._create_buffer(var_data_dict, cl_environment))
        data_buffers.extend(self._get_protocol_and_fixed_data_buffers(cl_environment))

        queue = self._get_queue(cl_environment)

        def eval_cb(x):
            cl.enqueue_copy(queue, param_buf, x.astype(np.float64, order='C'), is_blocking=True)
            kernel.evaluate(queue, (1, ), None, *data_buffers)
            cl.enqueue_copy(queue, errors, errors_buf, is_blocking=True)
            return errors[0]

        return eval_cb

    def _get_kernel(self, decode_params, var_data_dict, environment):
        nmr_params = self._state.model.get_nmr_estimable_parameters()

        prtcl_data_dict = self._state.prtcl_data_dict
        fixed_data_dict = self._state.fixed_data_dict

        param_code_gen = ParameterCLCodeGenerator(environment.device, var_data_dict, prtcl_data_dict,
                                                  fixed_data_dict, add_var_data_multipliers=False)

        kernel_param_names = ['constant double* params', 'global double* fval']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = get_cl_double_extension_definer(environment.platform)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += self._state.model.get_objective_function('calculateObjective')

        use_codec = False
        if decode_params:
            if self._state.model.get_parameter_codec():
                use_codec = True
                kernel_source += self._state.model.get_parameter_codec().get_cl_decode_function('decodeParameters')

        kernel_source += '''
            __kernel void evaluate(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''
                    double x[''' + repr(nmr_params) + '''];
                    for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        x[i] = params[i];
                    }
                    ''' + ('decodeParameters(x);' if use_codec else '') + '''

                    fval[0] = calculateObjective(&data, x);
                }
        '''
        warnings.simplefilter("ignore")
        return cl.Program(environment.context, kernel_source).build(' '.join(environment.compile_flags))


class _CodecCBGenerator(_BaseCBGenerator):

    def get_encode_cb(self, cl_environment):
        """Get the callback function for encoding parameters.

        Args:
            cl_environment (CLEnvironment): The cl environment to use,
                if none given we use the cl environment defined in the constructor.

        Returns:
            A python callback function that takes as parameter an numpy array of size (1, n) and that returns as output
             the encoded parameters.
        """
        return self._get_cb(cl_environment, 'encode')

    def get_decode_cb(self, cl_environment):
        """Get the callback function for decoding parameters.

        Args:
            cl_environment (CLEnvironment): The cl environment to use, if none given we
                use the cl environment defined in the constructor.

        Returns:
            A python callback function that takes as parameter an numpy array of size (1, n) and that returns as output
            the decoded parameters.
        """
        return self._get_cb(cl_environment, 'decode')

    def _get_cb(self, cl_environment, mode):
        if not cl_environment:
            cl_environment = self._state.cl_environment

        if cl_environment in self._state.cl_environment_items_cache:
            cl_items = self._state.cl_environment_items_cache[cl_environment]

            if mode == 'encode':
                if cl_items.param_encode_cb:
                    return cl_items.param_encode_cb
            else:
                if cl_items.param_decode_cb:
                    return cl_items.param_decode_cb
        else:
            self._state.cl_environment_items_cache.update({cl_environment: _CLEnvironmentsCachedItems()})

        codec = self._state.model.get_parameter_codec()

        if mode == 'encode':
            kernel = self._get_kernel(codec.get_cl_encode_function('encodeParameters'),
                                      'encodeParameters', codec.get_nmr_parameters(), cl_environment)
        else:
            kernel = self._get_kernel(codec.get_cl_decode_function('decodeParameters'),
                                      'decodeParameters', codec.get_nmr_parameters(), cl_environment)

        queue = self._get_queue(cl_environment)
        read_write_flags = cl_environment.get_read_write_cl_mem_flags()

        def cb(params_model_space):
            params = params_model_space.copy()
            param_buf = cl.Buffer(cl_environment.context, read_write_flags, hostbuf=params)
            kernel.transformParameterSpace(queue, (1, ), None, param_buf)
            cl.enqueue_copy(queue, params, param_buf, is_blocking=True)
            return params

        if mode == 'encode':
            self._state.cl_environment_items_cache[cl_environment].param_encode_cb = cb
        else:
            self._state.cl_environment_items_cache[cl_environment].param_decode_cb = cb

        return cb

    def _get_kernel(self, cl_func, cl_func_name, nmr_params, environment):
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
        warnings.simplefilter("ignore")
        return cl.Program(environment.context, kernel_source).build(' '.join(environment.compile_flags))


class _FinalTransformationCBGenerator(_BaseCBGenerator):

    def get_cb(self, cl_environment):
        """Get the final parameter transformations callback function.

        Args:
            cl_environment (CLEnvironment): The cl environment to use, if none given we use the
                cl environment globally defined in the constructor.

        Returns:
            A python callback function that takes as parameter an numpy array of size (1, n) and that returns as output
            an array of the same shape but with the parameters decoded to their final state.
        """
        if not cl_environment:
            cl_environment = self._state.cl_environment

        if cl_environment in self._state.cl_environment_items_cache:
            cl_items = self._state.cl_environment_items_cache[cl_environment]
            if cl_items.final_param_transform_cb:
                return cl_items.final_param_transform_cb
        else:
            self._state.cl_environment_items_cache.update({cl_environment: _CLEnvironmentsCachedItems()})

        prtcl_and_fixed_data_buffers = self._get_protocol_and_fixed_data_buffers(cl_environment)

        cl_transform_func = self._state.model.get_final_parameter_transformations('applyFinalParameterTransformations')

        if cl_transform_func:
            kernel = self._get_kernel(cl_transform_func, cl_environment)
            queue = self._get_queue(cl_environment)
            read_write_flags = cl_environment.get_read_write_cl_mem_flags()

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

        self._state.cl_environment_items_cache[cl_environment].final_param_transform_cb = final_param_transform_cb
        return final_param_transform_cb

    def _get_kernel(self, transform_func, environment):
        nmr_params = self._state.model.get_nmr_estimable_parameters()

        prtcl_data_dict = self._state.prtcl_data_dict
        fixed_data_dict = self._state.fixed_data_dict

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
        warnings.simplefilter("ignore")
        return cl.Program(environment.context, kernel_source).build(' '.join(environment.compile_flags))


class _LogPriorCBGenerator(_BaseCBGenerator):

    def get_cb(self, cl_environment):
        """Get the callback function for the log prior.

        Args:
            cl_environment (CLEnvironment): The cl environment to use, if none given we use the
                cl environment globally defined in the constructor.

        Returns:
            A python callback function that takes as parameter an numpy array of size (1, n) and that returns as output
            a single scalar value with the prior value for the given parameters.
        """
        if not cl_environment:
            cl_environment = self._state.cl_environment

        if cl_environment in self._state.cl_environment_items_cache:
            cl_items = self._state.cl_environment_items_cache[cl_environment]
            if cl_items.log_prior_cb:
                return cl_items.log_prior_cb
        else:
            self._state.cl_environment_items_cache.update({cl_environment: _CLEnvironmentsCachedItems()})

        queue = self._get_queue(cl_environment)
        read_only_flags = cl_environment.get_read_only_cl_mem_flags()
        write_only_flags = cl_environment.get_write_only_cl_mem_flags()

        result_buffer = cl.Buffer(cl_environment.context, write_only_flags, hostbuf=np.array((1,), dtype=np.float64))
        kernel = self._get_kernel(cl_environment)

        def log_prior_cb(params_model_space):
            param_buf = cl.Buffer(cl_environment.context, read_only_flags, hostbuf=params_model_space)
            kernel.calculateLogPrior(queue, (1, ), None, param_buf, result_buffer)
            prior_result = np.array((1,), dtype=np.float64)
            cl.enqueue_copy(queue, prior_result, result_buffer, is_blocking=True)
            return prior_result[0]

        self._state.cl_environment_items_cache[cl_environment].log_prior_cb = log_prior_cb
        return log_prior_cb

    def _get_kernel(self, cl_environment):
        nmr_params = self._state.model.get_nmr_estimable_parameters()
        kernel_source = get_cl_double_extension_definer(cl_environment.platform)
        kernel_source += self._state.model.get_log_prior_function('getLogPrior')
        kernel_source += '''
            __kernel void calculateLogPrior(constant double* x_global, global double* result){
                double x[''' + repr(nmr_params) + '''];
                for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                    x[i] = x_global[i];
                }
                result[0] = getLogPrior(x);
            }
        '''
        warnings.simplefilter("ignore")
        return cl.Program(cl_environment.context, kernel_source).build(' '.join(cl_environment.compile_flags))


class _ProposalCBGenerator(_BaseCBGenerator):

    def get_cb(self, cl_environment):
        """Get the callback function for the proposal function used in sampling.

        Args:
            cl_environment (CLEnvironment): The cl environment to use, if none given we use the
                cl environment globally defined in the constructor.

        Returns:
            A python callback function that takes as parameters:
                - the parameter index in the list of parameters for the current parameter we want the proposal of
                - the current value of that parameter.
        """
        if not cl_environment:
            cl_environment = self._state.cl_environment

        if cl_environment in self._state.cl_environment_items_cache:
            cl_items = self._state.cl_environment_items_cache[cl_environment]
            if cl_items.proposal_cb:
                return cl_items.proposal_cb
        else:
            self._state.cl_environment_items_cache.update({cl_environment: _CLEnvironmentsCachedItems()})

        queue = self._get_queue(cl_environment)
        write_only_flags = cl_environment.get_write_only_cl_mem_flags()

        ranluxcltab_buffer = initialize_ranlux(cl_environment, queue, 1, seed=1)
        result_buffer = cl.Buffer(cl_environment.context, write_only_flags, hostbuf=np.array((1,), dtype=np.float64))
        kernel = self._get_kernel(cl_environment)

        def proposal_cb(param_index, current_value):
            buffers = [np.int32(param_index),
                       np.float64(current_value),
                       ranluxcltab_buffer,
                       result_buffer]
            kernel.calculateProposal(queue, (1, ), None, *buffers)
            result_host = np.array((1,), dtype=np.float64)
            cl.enqueue_copy(queue, result_host, result_buffer, is_blocking=True)
            return result_host[0]

        self._state.cl_environment_items_cache[cl_environment].proposal_cb = proposal_cb
        return proposal_cb

    def _get_kernel(self, cl_environment):
        kernel_source = get_cl_double_extension_definer(cl_environment.platform)
        kernel_source += RanluxCL().get_cl_code()
        kernel_source += self._state.model.get_proposal_function('getProposal')
        kernel_source += '''
            __kernel void calculateProposal(const int parameter_index, const double current_value,
                                            global float4* ranluxcltab, global double* result){

                ranluxcl_state_t ranluxclstate;
                ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

                result[0] = getProposal(parameter_index, current_value, &ranluxclstate);

                ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
            }
        '''
        warnings.simplefilter("ignore")
        return cl.Program(cl_environment.context, kernel_source).build(' '.join(cl_environment.compile_flags))