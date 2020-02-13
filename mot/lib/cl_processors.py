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

    def process(self):
        """Enqueue all compute kernels for this processor.

        This may enqueue multiple kernels to multiple devices.
        """
        raise NotImplementedError()

    def flush(self):
        """Enqueues a flush operation to all the queues."""
        raise NotImplementedError()

    def finish(self):
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

    def process(self):
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

    def flush(self):
        self._cl_environment.queue.flush()

    def finish(self):
        self._cl_environment.queue.finish()

    def _flatten_list(self, l):
        return_l = []
        for e in l:
            return_l.extend(e)
        return return_l


class CLFunctionProcessor(Processor):

    def __init__(self, kernels, kernel_data, nmr_instances, use_local_reduction=False,
                 local_size=None, cl_runtime_info=None, do_data_transfers=True):
        """Create a processor for the given function and inputs.

        The typical way of using this processor is by:
        1) create it
        2) use :meth:`get_kernel_data` to get the kernel data elements and use get_data() for each of them to get the
            underlying data. You can then modify that.
        3) call :meth:`process` to run this function on all devices with the current data
        4) call :meth:`finish` to finish the execution (or do not call it and chain another operation)
        5) optionally use :meth:`get_function_results` to get the function results (if the function had a non-void
            return signature).

        Args:
            kernel_data (Iterable[Union(ndarray, mot.lib.utils.KernelData)]
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
            do_data_transfers (boolean): if we should do data transfers from host to device and back for evaluating
                this function. For better control set this to False and use the method
                ``enqueue_device_access()`` and ``enqueue_host_access`` of the KernelData to set the data.
        """
        self._cl_runtime_info = cl_runtime_info or CLRuntimeInfo()
        self._cl_environments = self._cl_runtime_info.cl_environments
        self._batches = self._cl_runtime_info.load_balancer.get_division(self._cl_environments, nmr_instances)

        self._subprocessors = []
        for ind, cl_environment in enumerate(self._cl_environments):
            kernel = kernels[cl_environment]

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
                processor = SimpleProcessor(kernel, kernel_data.values(), cl_environment,
                                            batch_end - batch_start, workgroup_size,
                                            instance_offset=batch_start, do_data_transfers=do_data_transfers)
                self._subprocessors.append(processor)

    def process(self):
        for worker in self._subprocessors:
            worker.process()
            worker.flush()

    def flush(self):
        for worker in self._subprocessors:
            worker.flush()

    def finish(self):
        for worker in self._subprocessors:
            worker.finish()
