__author__ = 'Robbert Harms'
__date__ = '2020-01-25'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'

from collections import OrderedDict

import numpy as np
import pyopencl as cl

from mot.configuration import CLRuntimeInfo
from mot.lib.kernel_data import Zeros
from mot.lib.utils import convert_inputs_to_kernel_data, get_cl_utility_definitions


class Processor:

    def __init__(self, cl_function, inputs, nmr_instances, use_local_reduction=False,
                 local_size=None, cl_runtime_info=None):
        """Create a processor for the given function and inputs.

        By changing the data underlying the kernel data elements, this processor allows for multiple executions
        with only a low overhead.

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
            cl_runtime_info (mot.configuration.CLRuntimeInfo): the runtime information for execution
        """
        self._cl_function = cl_function
        self._kernel_data = convert_inputs_to_kernel_data(inputs, cl_function.get_parameters(), nmr_instances)

        if cl_function.get_return_type() != 'void':
            self._kernel_data['_results'] = Zeros((nmr_instances,), cl_function.get_return_type())

        cl_runtime_info = cl_runtime_info or CLRuntimeInfo()
        cl_environments = cl_runtime_info.cl_environments

        mot_float_dtype = np.float32
        if cl_runtime_info.double_precision:
            mot_float_dtype = np.float64

        for data in self._kernel_data.values():
            data.set_mot_float_dtype(mot_float_dtype)

        self._workers = []
        for ind, cl_environment in enumerate(cl_environments):
            worker = KernelWorker(cl_function, self._kernel_data, cl_environment,
                                  compile_flags=cl_runtime_info.compile_flags,
                                  double_precision=cl_runtime_info.double_precision,
                                  use_local_reduction=use_local_reduction,
                                  local_size=local_size)
            self._workers.append(worker)
        self._batches = cl_runtime_info.load_balancer.get_division(cl_environments, nmr_instances)

    def enqueue_run(self, flush=True, finish=False):
        """Enqueues and flushes running this function on all devices.

        Args:
            flush (boolean): if we flush the queues after enqueuing the work
            finish (boolean): if enqueue a finish operation to the queues after enqueueing the work
        """
        for worker, (batch_start, batch_end) in zip(self._workers, self._batches):
            if batch_end - batch_start > 0:
                worker.calculate(batch_start, batch_end)

                if flush:
                    worker.cl_queue.flush()
        if finish:
            self.enqueue_finish()

    def enqueue_finish(self):
        """Enqueues a finish operation to all the queues."""
        for worker in self._workers:
            worker.cl_queue.finish()

    def get_kernel_data(self):
        """Get the kernel data object this processor is working on"""
        return self._kernel_data

    def get_function_results(self):
        """Get the current function results. Only useful if the function has a non-void return type.

        Returns:
            ndarray: the return values of the function, which can be None if this function has a void return type.
        """
        if self._cl_function.get_return_type() != 'void':
            return self._kernel_data['_results'].get_data()


class KernelWorker:

    def __init__(self, cl_function, kernel_data, cl_environment, compile_flags=None,
                 double_precision=False, use_local_reduction=False, local_size=None):
        """Create a processor able to process the given function with the given data in the given environment.

        Objects of this type can be used in pipelines since very fast execution can be achieved by creating it once
        and then changing the underlying data of the kernel data objects.
        """
        self._cl_environment = cl_environment
        self._cl_context = cl_environment.context
        self._cl_queue = cl_environment.queue
        self._kernel_data = OrderedDict(sorted(kernel_data.items()))
        self._double_precision = double_precision
        self._use_local_reduction = use_local_reduction

        if cl_function.is_kernel_func():
            self._cl_function_kernel = cl_function
        else:
            self._cl_function_kernel = cl_function.created_wrapped_kernel_func(self._kernel_data)

        self._kernel = cl.Program(self._cl_context, self._get_kernel_source()).build(' '.join(compile_flags))
        self._kernel_func = getattr(self._kernel, self._cl_function_kernel.get_cl_function_name())

        if self._use_local_reduction:
            if local_size:
                self._workgroup_size = local_size
            else:
                self._workgroup_size = self._kernel_func.get_work_group_info(
                    cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                    self._cl_environment.device)
        else:
            self._workgroup_size = 1

        self._kernel_inputs = {name: data.get_kernel_inputs(self._cl_context, self._workgroup_size)
                               for name, data in self._kernel_data.items()}
        self._kernel_func.set_scalar_arg_dtypes(self.get_scalar_arg_dtypes())

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

        kernel_inputs_list = []
        for inputs in [self._kernel_inputs[name] for name in self._kernel_data]:
            kernel_inputs_list.extend(inputs)

        for name, data in self._kernel_data.items():
            data.enqueue_device_access(self._cl_queue, self._kernel_inputs[name], range_start, range_end)

        self._kernel_func(
            self._cl_queue,
            (int(nmr_problems * self._workgroup_size),),
            (int(self._workgroup_size),),
            *kernel_inputs_list,
            global_offset=(int(range_start * self._workgroup_size),))

        for name, data in self._kernel_data.items():
            data.enqueue_host_access(self._cl_queue, self._kernel_inputs[name], range_start, range_end)

    def _get_kernel_source(self):
        kernel_source = ''
        kernel_source += get_cl_utility_definitions(self._double_precision)
        kernel_source += '\n'.join(data.get_type_definitions() for data in self._kernel_data.values())
        kernel_source += self._cl_function_kernel.get_cl_code()
        return kernel_source

    def get_scalar_arg_dtypes(self):
        """Get the location and types of the input scalars.

        Returns:
            list: for every kernel input element either None if the data is a buffer or the numpy data type if
                if is a scalar.
        """
        dtypes = []
        for name, data in self._kernel_data.items():
            dtypes.extend(data.get_scalar_arg_dtypes())
        return dtypes
