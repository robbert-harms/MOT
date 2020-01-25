__author__ = 'Robbert Harms'
__date__ = '2020-01-25'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'

import os
from collections import OrderedDict
import numpy as np
import pyopencl as cl
from pkg_resources import resource_filename
from mot.configuration import CLRuntimeInfo
from mot.lib.kernel_data import Array, Zeros
from mot.lib.utils import convert_inputs_to_kernel_data, get_cl_utility_definitions


class Processor:

    def enqueue_kernels(self, flush=True, finish=False):
        """Enqueue all compute kernels for this processor.

        This may enqueue multiple kernels to multiple devices. It may call flush and finish afterwards if set.

        Args:
            flush (boolean): if we flush the queues after enqueuing the kernels
            finish (boolean): if we enqueue a finish operation after enqueueing the kernels
        """
        raise NotImplementedError()

    def enqueue_finish(self):
        """Enqueues a finish operation to all the queues."""
        raise NotImplementedError()

    def get_kernel_data(self):
        """Get the kernel data this processor is working on.

        Returns:
            dict[str: mot.lib.kernel_data.KernelData]: dictionary mapping parameters to kernel data elements.
        """
        raise NotImplementedError()


class CLFunctionProcessor(Processor):

    def __init__(self, cl_function, inputs, nmr_instances, use_local_reduction=False, local_size=None,
                 context_variables=None, enable_rng=False, cl_runtime_info=None):
        """Create a processor for the given function and inputs.

        The typical way of using this processor is by:
        1) create it
        2) use :meth:`get_kernel_data` to get the kernel data elements and use get_data() for each of them to get the
            underlying data. You can then modify that.
        3) call :meth:`enqueue_run` to run this function on all devices with the current data
        4) call :meth:`enqueue_finish` to finish the execution (or do not call it and chain another operation)
        5) optionally use :meth:`get_function_results` to get the function results (if the function had a non-void
            return signature).

        Args:
            inputs (Iterable[Union(ndarray, mot.lib.utils.KernelData)]
                    or Mapping[str: Union(ndarray, mot.lib.utils.KernelData)]): for each CL function parameter
                the input data. Each of these input datasets must either be a scalar or be of equal length in the
                first dimension. The elements can either be raw ndarrays or KernelData objects.
                If an ndarray is given we will load it read/write by default. You can provide either an iterable
                with one value per parameter, or a mapping with for every parameter a corresponding value.
            nmr_instances (int): the number of parallel processes to run.
            use_local_reduction (boolean): set this to True if you want to use local memory reduction in
                 evaluating this function. If this is set to True we will multiply the global size
                 (given by the nmr_instances) by the work group sizes.
            local_size (int): can be used to specify the exact local size (workgroup size) the kernel must use.
            context_variables (dict[str: mot.lib.kernel_data.KernelData]): data structures that will be loaded
                as program scope global variables. Note that not all KernelData types are allowed, only the
                global variables are allowed.
            enable_rng (boolean): if this function wants to use random numbers. If set to true we prepare the random
                number generator for use in this function.
            cl_runtime_info (mot.configuration.CLRuntimeInfo): the runtime information for execution
        """
        self._original_cl_function = cl_function

        self._cl_runtime_info = cl_runtime_info or CLRuntimeInfo()
        self._cl_environments = self._cl_runtime_info.cl_environments
        self._cl_function, self._kernel_data = self._resolve_cl_function_and_kernel_data(
            cl_function, inputs, nmr_instances)
        self._batches = self._cl_runtime_info.load_balancer.get_division(self._cl_environments, nmr_instances)

        context_variables = context_variables or {}
        if enable_rng:
            rng_state = np.random.uniform(low=np.iinfo(np.uint32).min, high=np.iinfo(np.uint32).max + 1,
                                          size=(nmr_instances, 4)).astype(np.uint32)
            context_variables['__rng_state'] = Array(rng_state, 'uint', mode='rw', ensure_zero_copy=True)

        mot_float_dtype = np.float32
        if self._cl_runtime_info.double_precision:
            mot_float_dtype = np.float64

        for data in context_variables.values():
            data.set_mot_float_dtype(mot_float_dtype)

        self._workers = []
        for ind, cl_environment in enumerate(self._cl_environments):
            worker = _KernelWorker(self._cl_function, self._kernel_data, cl_environment,
                                   compile_flags=self._cl_runtime_info.compile_flags,
                                   double_precision=self._cl_runtime_info.double_precision,
                                   use_local_reduction=use_local_reduction,
                                   local_size=local_size,
                                   enable_rng=enable_rng, context_variables=context_variables)
            self._workers.append(worker)

    def enqueue_kernels(self, flush=True, finish=False):
        for worker, (batch_start, batch_end) in zip(self._workers, self._batches):
            if batch_end - batch_start > 0:
                worker.calculate(batch_start, batch_end)

                if flush:
                    worker.cl_queue.flush()
        if finish:
            self.enqueue_finish()

    def enqueue_finish(self):
        for worker in self._workers:
            worker.cl_queue.finish()

    def get_kernel_data(self):
        return self._kernel_data

    def get_function_results(self):
        """Get the current function results. Only useful if the function has a non-void return type.

        Returns:
            ndarray: the return values of the function, which can be None if this function has a void return type.
        """
        if self._original_cl_function.get_return_type() != 'void':
            return self._kernel_data['_return_values'].get_data()

    def _resolve_cl_function_and_kernel_data(self, cl_function, inputs, nmr_instances):
        """Ensures that the CLFunction is a kernel function and the inputs are kernel data elements."""
        kernel_data = convert_inputs_to_kernel_data(inputs, cl_function.get_parameters(), nmr_instances)

        if cl_function.get_return_type() != 'void':
            kernel_data['_return_values'] = Zeros((nmr_instances,), cl_function.get_return_type())

        mot_float_dtype = np.float32
        if self._cl_runtime_info.double_precision:
            mot_float_dtype = np.float64

        for data in kernel_data.values():
            data.set_mot_float_dtype(mot_float_dtype)

        kernel_data = OrderedDict(sorted(kernel_data.items()))

        if not cl_function.is_kernel_func():
            cl_function = cl_function.created_wrapped_kernel_func(kernel_data)

        return cl_function, kernel_data


class _KernelWorker:

    def __init__(self, cl_function, kernel_data, cl_environment, compile_flags=None,
                 double_precision=False, use_local_reduction=False, local_size=None,
                 context_variables=None, enable_rng=False):
        """Create a processor able to process the given function with the given data in the given environment.

        Objects of this type can be used in pipelines since very fast execution can be achieved by creating it once
        and then changing the underlying data of the kernel data objects.
        """
        self._cl_context = cl_environment.context
        self._cl_queue = cl_environment.queue
        self._kernel_data = kernel_data

        context_variables = context_variables or {}
        self._context_variables = OrderedDict(sorted(context_variables.items()))

        kernel_source = self._get_kernel_source(cl_function, kernel_data,
                                                context_variables, double_precision, enable_rng)
        self._program = cl.Program(cl_environment.context, kernel_source).build(' '.join(compile_flags))
        self._kernel = getattr(self._program, cl_function.get_cl_function_name())

        if use_local_reduction:
            if local_size:
                self._workgroup_size = local_size
            else:
                self._workgroup_size = self._kernel.get_work_group_info(
                    cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, cl_environment.device)
        else:
            self._workgroup_size = 1

    @property
    def cl_queue(self):
        """Get the queue this worker is using for its GPU computations.

        This may be used to flush or finish the queue to provide synchronization.

        Returns:
            pyopencl queue: the queue used by this worker
        """
        return self._cl_queue

    def calculate(self, range_start, range_end):
        """Start processing the current data on the given range.

        Args:
            range_start (int): the beginning of the range we will process (defines the start of the global offset)
            range_end (int): the end of the range we will process
        """
        nmr_problems = range_end - range_start

        def run_context_var_kernel():
            kernel_inputs = {}
            kernel_inputs_order = []
            for name, data in self._context_variables.items():
                kernel_inputs[name] = data.get_kernel_inputs(self._cl_context, self._workgroup_size)
                kernel_inputs_order.append(name)

            kernel_inputs_list = []
            for inputs in [kernel_inputs[name] for name in kernel_inputs_order]:
                kernel_inputs_list.extend(inputs)

            for name, data in self._context_variables.items():
                data.enqueue_device_access(self._cl_queue, kernel_inputs[name], range_start, range_end)

            func = self._program._initialize_context_variables
            func(
                self._cl_queue,
                (int(nmr_problems * self._workgroup_size),),
                (int(self._workgroup_size),),
                *kernel_inputs_list,
                global_offset=(int(range_start * self._workgroup_size),))

            for name, data in self._context_variables.items():
                data.enqueue_host_access(self._cl_queue, kernel_inputs[name], range_start, range_end)

        def run_compute_kernel():
            kernel_inputs = {}
            kernel_inputs_order = []
            for name, data in self._kernel_data.items():
                kernel_inputs[name] = data.get_kernel_inputs(self._cl_context, self._workgroup_size)
                kernel_inputs_order.append(name)

            kernel_inputs_list = []
            for inputs in [kernel_inputs[name] for name in kernel_inputs_order]:
                kernel_inputs_list.extend(inputs)

            for name, data in self._kernel_data.items():
                data.enqueue_device_access(self._cl_queue, kernel_inputs[name], range_start, range_end)

            self._kernel(
                self._cl_queue,
                (int(nmr_problems * self._workgroup_size),),
                (int(self._workgroup_size),),
                *kernel_inputs_list,
                global_offset=(int(range_start * self._workgroup_size),))

            for name, data in self._kernel_data.items():
                data.enqueue_host_access(self._cl_queue, kernel_inputs[name], range_start, range_end)

        if self._context_variables:
            run_context_var_kernel()
        run_compute_kernel()

    def _get_kernel_source(self, cl_function, kernel_data, context_variables, double_precision, enable_rng):
        kernel_source = ''
        kernel_source += get_cl_utility_definitions(double_precision)
        kernel_source += '\n'.join(data.get_type_definitions() for data in kernel_data.values())
        kernel_source += '\n'.join(data.get_type_definitions() for data in context_variables.values())
        kernel_source += '\n'.join(data.get_context_variable_declaration(name)
                                   for name, data in context_variables.items())
        if enable_rng:
            kernel_source += self._get_rng_cl_code()

        if context_variables:
            kernel_source += self._get_context_variable_init_function(context_variables).get_cl_code()

        kernel_source += cl_function.get_cl_code()
        return kernel_source

    def _get_context_variable_init_function(self, context_variables):
        from mot.lib.cl_function import SimpleCLFunction

        kernel_name = '_initialize_context_variables'

        parameter_list = []
        context_inits = []
        for name, data in context_variables.items():
            parameter_list.extend(data.get_kernel_parameters('_context_' + name))
            context_inits.append(data.get_context_variable_initialization(name, '_context_' + name))

        cl_body = '''
            ulong gid = (ulong)(get_global_id(0) / get_local_size(0));
            ''' + '\n'.join(context_inits) + '''
        '''
        return SimpleCLFunction('void', kernel_name, parameter_list, cl_body, is_kernel_func=True)

    def _get_rng_cl_code(self):
        generator = 'philox'

        src = open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/openclfeatures.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/array.h'), ), 'r').read()
        src += open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/{}.h'.format(generator)), ),
                    'r').read()
        src += (open(os.path.abspath(resource_filename('mot', 'data/opencl/random123/rand123.h'), ), 'r').read() % {
            'GENERATOR_NAME': generator})

        return src
