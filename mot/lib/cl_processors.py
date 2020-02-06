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

    def enqueue_process(self, flush=True, finish=False):
        """Enqueue all compute kernels for this processor.

        This may enqueue multiple kernels to multiple devices. It may call flush and finish afterwards if set.

        Args:
            flush (boolean): if we flush the queues after enqueuing the kernels
            finish (boolean): if we enqueue a finish operation after enqueueing the kernels
        """
        raise NotImplementedError()

    def enqueue_flush(self):
        """Enqueues a flush operation to all the queues."""
        raise NotImplementedError()

    def enqueue_finish(self):
        """Enqueues a finish operation to all the queues."""
        raise NotImplementedError()


class SimpleProcessor(Processor):

    def __init__(self, kernel, kernel_data, cl_environment, global_nmr_instances, workgroup_size, instance_offset=None,
                 do_data_transfers=True):
        """Simple processor which can execute the provided (compiled) kernel with the provided data.

        Args:
            kernel: a pyopencl compiled kernel program
            kernel_data (List[mot.lib.utils.KernelData]): the kernel data to load as input to the kernel
            cl_environment (mot.lib.cl_environments.CLEnvironment): the CL environment to use for executing the kernel
            global_nmr_instances (int): the global work size, this will internally be multiplied by the
                local workgroup size.
            workgroup_size (int): the local size (workgroup size) the kernel must use
            instance_offset (int): the offset for the global id, this will be multiplied with the local workgroup size.
            do_data_transfers (boolean): if this processor should do the data transfers for the kernel data elements.
                If set to True, this will call ``enqueue_device_access`` and ``enqueue_host_access`` on each kernel
                data element.
        """
        self._kernel = kernel
        self._kernel_data = kernel_data
        self._cl_environment = cl_environment
        self._global_nmr_instances = global_nmr_instances
        self._instance_offset = instance_offset or 0
        self._kernel.set_scalar_arg_dtypes(self._flatten_list([d.get_scalar_arg_dtypes() for d in self._kernel_data]))
        self._workgroup_size = workgroup_size
        self._do_data_transfers = do_data_transfers

    def enqueue_process(self, flush=True, finish=False):
        kernel_inputs = [data.get_kernel_inputs(self._cl_environment, self._workgroup_size)
                         for data in self._kernel_data]

        if self._do_data_transfers:
            for ind, kernel_data in enumerate(self._kernel_data):
                kernel_data.enqueue_device_access(self._cl_environment)

        self._kernel(
            self._cl_environment.queue,
            (int(self._global_nmr_instances * self._workgroup_size),),
            (int(self._workgroup_size),),
            *self._flatten_list(kernel_inputs),
            global_offset=(int(self._instance_offset * self._workgroup_size),))

        if self._do_data_transfers:
            for ind, kernel_data in enumerate(self._kernel_data):
                kernel_data.enqueue_host_access(self._cl_environment)

        if flush:
            self.enqueue_flush()
        if finish:
            self.enqueue_finish()

    def enqueue_flush(self):
        self._cl_environment.queue.flush()

    def enqueue_finish(self):
        self._cl_environment.queue.finish()

    def _flatten_list(self, l):
        return_l = []
        for e in l:
            return_l.extend(e)
        return return_l


class CLFunctionProcessor(Processor):

    def __init__(self, cl_function, inputs, nmr_instances, use_local_reduction=False,
                 local_size=None, cl_runtime_info=None):
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
            cl_runtime_info (mot.configuration.CLRuntimeInfo): the runtime information for execution
        """
        self._original_cl_function = cl_function

        self._cl_runtime_info = cl_runtime_info or CLRuntimeInfo()
        self._cl_environments = self._cl_runtime_info.cl_environments
        self._cl_function, self._kernel_data = self._resolve_cl_function_and_kernel_data(
            cl_function, inputs, nmr_instances)
        self._batches = self._cl_runtime_info.load_balancer.get_division(self._cl_environments, nmr_instances)

        self._subprocessors = []
        for ind, cl_environment in enumerate(self._cl_environments):
            program = cl.Program(cl_environment.context, self._get_kernel_source()).build(
                ' '.join(self._cl_runtime_info.compile_flags))
            kernel = getattr(program, self._cl_function.get_cl_function_name())

            if use_local_reduction:
                if local_size:
                    workgroup_size = local_size
                else:
                    workgroup_size = kernel.get_work_group_info(
                        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, cl_environment.device)
            else:
                workgroup_size = 1

            batch_start, batch_end = self._batches[ind]

            if batch_end - batch_start > 0:
                if batch_end - batch_start == nmr_instances:
                    kernel_data = self._kernel_data.values()
                else:
                    kernel_data = [v.get_subset(batch_range=self._batches[ind]) for v in self._kernel_data.values()]
                processor = SimpleProcessor(kernel, kernel_data, cl_environment,
                                            batch_end - batch_start, workgroup_size)
                self._subprocessors.append(processor)

    def enqueue_process(self, flush=True, finish=False):
        for worker in self._subprocessors:
            worker.enqueue_process(flush=flush, finish=finish)

    def enqueue_flush(self):
        for worker in self._subprocessors:
            worker.enqueue_flush()

    def enqueue_finish(self):
        for worker in self._subprocessors:
            worker.enqueue_finish()

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

    def _get_kernel_source(self):
        kernel_source = ''
        kernel_source += get_cl_utility_definitions(self._cl_runtime_info.double_precision)
        kernel_source += '\n'.join(data.get_type_definitions() for data in self._kernel_data.values())
        kernel_source += self._cl_function.get_cl_code()
        return kernel_source
