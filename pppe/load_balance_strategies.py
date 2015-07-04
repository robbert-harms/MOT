import logging
import math
import time
import warnings
import numpy as np
import pyopencl as cl
from .utils import get_read_only_cl_mem_flags, device_type_from_string


__author__ = 'Robbert Harms'
__date__ = "2014-06-23"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Worker(object):

    def __init__(self, cl_environment):
        """Create a new worker object. This is meant to be subclassed by the user.

        The idea is that the workload strategy can use this object to calculate the data, in a way it seems
        fit for the strategy. During determining the strategy, all items computed should be stored
        internally by the worker.

        Args:
            cl_environment (CLEnvironment): The cl environment, can be used to determine the load
        """
        self._cl_environment = cl_environment
        self._queue = self._cl_environment.get_new_queue()
        self._constant_buffers = []

    @property
    def cl_environment(self):
        """Get the used CL environment.

        Returns:
            cl_environment (CLEnvironment): The cl environment to use for calculations.
        """
        return self._cl_environment

    def calculate(self, range_start, range_end):
        """Calculate for this problem the given range.

        The results of the computations must be stored internally.

        Args:
            range_start (int): The start of the range
            range_end (int): The end of the range

        Returns:
            cl_event: The last CL event, so the load balancer can wait for completion on it.
        """

    def _build_kernel(self):
        """Build the kernel for this worker.

        This assumes that the implementer implements the function _get_kernel_source() to get the source.

        Returns:
            a compiled kernel
        """
        kernel_source = self._get_kernel_source()
        warnings.simplefilter("ignore")
        kernel = cl.Program(self._cl_environment.context,
                            kernel_source).build(' '.join(self._cl_environment.compile_flags))
        return kernel

    def _get_kernel_source(self):
        """Calculate the kernel source for this worker.

        Returns:
            str: the kernel
        """

    def _generate_constant_buffers(self, *args):
        """Generate read only buffers for the given data

        Args:
            args (list of dicts): The list with dictionaries with the values we want to buffer.

        Returns:
            a list of the same length with read only cl buffers.
        """
        buffers = []
        for data_dict in args:
            for data in data_dict.values():
                if isinstance(data, np.ndarray):
                    buffers.append(cl.Buffer(self._cl_environment.context,
                                             get_read_only_cl_mem_flags(self._cl_environment),
                                             hostbuf=data))
                else:
                    buffers.append(data)
        return buffers


class LoadBalanceStrategy(object):

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def process(self, workers, nmr_items):
        """Process all of the items using the callback function in the work packages.

        The idea is that a strategy can be chosen on the fly by for example testing the execution time of the callback
        functions. Alternatively, a strategy can be determined based on the available environments (in the WorkPackages)
        and/or by the total number of items to be processed.

        Args:
            workers (Worker): a list of workers
            nmr_items (int): an integer specifying the total number of items to be processed
        """
        pass

    def get_used_cl_environments(self, cl_environments):
        """Get a subset of CL environments that this strategy plans on using.

        The CL routine contains the list of CL environments, which it gives to this function. It then
        expects back either same list or a proper subset of this list.

        This can be used by the using class to only create workers for environments actually in use. This might save
        compile time.

        Args:
            cl_environments: the CL environments we want this strategy to check if it wants to use them.

        Returns:
            A subset of the CL environments, can be all of them.
        """
        pass

    def _run_batches(self, workers, batches):
        """Run a list of batches on each of the workers.

        This will enqueue on all the workers the batches in sequence and waits for completion of each batch before
        enqueueing the next one.

        Args:
            workers (list of Worker): the workers to use in the processing
            batches (list of lists): for each worker a list with the batches in format (start, end)
        """
        self._logger.debug('Preparing to run {0} batch(es) on {1} device(s)'.format(len(batches[0]), len(workers)))

        for batch_nmr in range(len(batches[0])):
            self._logger.debug('Going to run batch {0} with range {1}'.format(batch_nmr, batches[0][batch_nmr]))

            events = []
            for worker_ind, worker in enumerate(workers):
                events.append(worker.calculate(*batches[worker_ind][batch_nmr]))
            for event in events:
                event.wait()

            self._logger.info('Processing at {}%.'.format(100 * (float(batch_nmr+1) / len(batches[0]))))

        self._logger.debug('Ran all batches')

    def _try_processing(self, worker, range_start, range_end):
        """Try to process the given worker on the given range.

        If processing fails due to memory problems we try to run the worker again with a smaller range.
        This is currently a blocking call if we run into a memory exception.

        Args:
            worker (Worker): The worker to use for the work
            range_start (int): the start of the range to process
            range_end (int): the end of the range to process
            wait_for_cl_event (CL Event): the CL event we must add to the kernel call. This is needed to support
                running multiple events nicely after each other with very large buffers.

        Returns:
            a cl event for the last event to happen. Unfortunately this is at the moment a blocking call if the
            worker throws a memory error. In the future this should be changed to something more appropriate.
        """
        try:
            return worker.calculate(range_start, range_end)
        except cl.MemoryError:
            half_range_length = int(math.ceil((range_end - range_start) / 2.0))
            event = self._try_processing(worker, range_start, range_start + half_range_length)
            event.wait()
            return self._try_processing(worker, range_start + half_range_length, range_end)


class EvenDistribution(LoadBalanceStrategy):

    def __init__(self, run_in_batches=True, single_batch_length=1e4):
        """Give each worker exactly 1/nth of the work. This does not do any feedback load balancing.

        Args:
            run_in_batches (boolean): If we want to run the load per worker in batches or in one large run.
                The advantage of batches is that it is interruptable and it may prevent memory errors since we run
                with smaller buffers. The disadvantage is that it may be slower due to constant waiting to load the
                new kernel.
            single_batch_length (int): The length of a single batch, only used if run_in_batches is set to True.
                This will create batches this size and run each of them one after the other.
        """
        super(EvenDistribution, self).__init__()
        self.run_in_batches = run_in_batches
        self.single_batch_length = single_batch_length

    def process(self, workers, nmr_items):
        items_per_worker = round(nmr_items / float(len(workers)))
        batches = []
        current_pos = 0

        for worker_ind in range(len(workers)):
            if worker_ind == len(workers) - 1:
                batches.append(self._create_batches(current_pos, nmr_items))
            else:
                batches.append(self._create_batches(current_pos, current_pos + items_per_worker))
                current_pos += items_per_worker

        self._run_batches(workers, batches)

    def get_used_cl_environments(self, cl_environments):
        return cl_environments

    def _create_batches(self, range_start, range_end):
        """Created batches in the given range.

        If self.run_in_batches is False we will only return one batch covering the entire range. If self.run_in_batches
        is True we will create batches the size of self.single_batch_length.

        Args:
            range_start (int): the start of the range to create batches for
            range_end (int): the end of the range to create batches for

        Returns:
            list of batches which are (start, end) pairs
        """

        if self.run_in_batches:
            batches = []
            batch_pos = range_start

            while batch_pos < range_end:
                new_batch = (batch_pos, min(range_end, batch_pos + self.single_batch_length))
                batches.append(new_batch)
                batch_pos = new_batch[1]

            return batches
        return [(range_start, range_end)]


class RuntimeLoadBalancing(LoadBalanceStrategy):

    def __init__(self, test_percentage=10):
        """Distribute the work by trying to minimize the time taken.

        Args:
            test_percentage (float): The percentage of items to use for the run time duration test
                (divided by number of devices)
        """
        super(RuntimeLoadBalancing, self).__init__()
        self.test_percentage = test_percentage

    def process(self, workers, nmr_items):
        durations = []
        start = 0
        for worker in workers:
            end = start + int(math.floor(nmr_items * (self.test_percentage/len(workers)) / 100))
            durations.append(self._test_duration(worker, start, end))
            start = end

        total_d = sum(durations)
        nmr_items_left = nmr_items - start

        finish_events = []
        for i in range(len(workers)):
            if i == len(workers) - 1:
                new_event = self._try_processing(workers[i], start, nmr_items)
            else:
                items = int(math.floor(nmr_items_left * (1 - (durations[i] / total_d))))
                new_event = self._try_processing(workers[i], start, start + items)
                start += items

            finish_events.append(new_event)

        for finish_event in finish_events:
            if finish_event:
                finish_event.wait()

    def _test_duration(self, worker, start, end):
        s = time.time()
        event = worker.calculate(start, end)
        if event:
            event.wait()
        return time.time() - s

    def get_used_cl_environments(self, cl_environments):
        return cl_environments


class PreferSingleDeviceType(LoadBalanceStrategy):

    def __init__(self, device_type=None, lb_strategy=None):
        """This is a meta load balance strategy, it uses the given strategy and prefers the use of the indicated device.

        Args:
            device_type (str or cl.device_type): either a cl device type or a string like ('gpu', 'cpu' or 'apu').
                This variable indicates the type of device we want to use.
            lb_strategy (LoadBalanceStrategy): The strategy this class uses in the background.
        """
        super(PreferSingleDeviceType, self).__init__()
        self._lb_strategy = lb_strategy or EvenDistribution()
        self._device_type = device_type or cl.device_type.CPU

        if isinstance(device_type, basestring):
            self._device_type = device_type_from_string(device_type)

    def process(self, workers, nmr_items):
        specific_workers = [worker for worker in workers if worker.cl_environment.device_type == self._device_type]

        if specific_workers:
            self._lb_strategy.process(specific_workers, nmr_items)
        else:
            self._lb_strategy.process(workers, nmr_items)

    def get_used_cl_environments(self, cl_environments):
        specific_envs = [cl_env for cl_env in cl_environments if cl_env.device_type == self._device_type]

        if specific_envs:
            return specific_envs
        else:
            return cl_environments


class PreferGPU(PreferSingleDeviceType):

    def __init__(self, lb_strategy=None):
        """This is a meta load balance strategy, it uses the given strategy and prefers the use of GPU's.

        Args:
            lb_strategy (LoadBalanceStrategy): The strategy this class uses in the background.
        """
        super(PreferGPU, self).__init__(device_type='GPU', lb_strategy=lb_strategy)


class PreferCPU(PreferSingleDeviceType):

    def __init__(self, lb_strategy=None):
        """This is a meta load balance strategy, it uses the given strategy and prefers the use of CPU's.

        Args:
            lb_strategy (LoadBalanceStrategy): The strategy this class uses in the background.
        """
        super(PreferCPU, self).__init__(device_type='CPU', lb_strategy=lb_strategy)