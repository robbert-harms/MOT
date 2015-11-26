import logging
import math
import time
import timeit
import warnings
import numpy as np
import pyopencl as cl
from six import string_types
from .utils import device_type_from_string


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
        self._cl_context = self._cl_environment.get_new_context()
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

        # The following is a hack to be able to build kernels using a nvidia 1.1 opencl implementation
        # using a 1.2 build PyOpenCL library and the Intel Opencl 1.2 driver loader.
        # todo remove this as soon as OpenCL 1.1 is no longer supported
        cl._DEFAULT_INCLUDE_OPTIONS = []

        kernel = cl.Program(self._cl_context.context,
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
                    buffers.append(cl.Buffer(self._cl_context.context,
                                             self._cl_environment.get_read_only_cl_mem_flags(),
                                             hostbuf=data))
                else:
                    buffers.append(data)
        return buffers


class LoadBalanceStrategy(object):

    def __init__(self, run_in_batches=True, single_batch_length=1e5):
        """ The base load balancer.

        Every load balancer has the option to run the calculations in batches. The advantage of batches is that it is
        interruptable and it may prevent memory errors since we run with smaller buffers. The disadvantage is that it
        may be slower due to constant waiting to load the new kernel and due to GPU thread starvation.

        Args:
            run_in_batches (boolean): If we want to run the load per worker in batches or in one large run.
            single_batch_length (int): The length of a single batch, only used if run_in_batches is set to True.
                This will create batches this size and run each of them one after the other.

        Attributes:
            run_in_batches (boolean); See above.
            single_batch_length (boolean); See above.
        """
        self._logger = logging.getLogger(__name__)
        self._run_in_batches = run_in_batches
        self._single_batch_length = single_batch_length

    @property
    def run_in_batches(self):
        return self._run_in_batches

    @run_in_batches.setter
    def run_in_batches(self, value):
        self._run_in_batches = value

    @property
    def single_batch_length(self):
        return self._single_batch_length

    @single_batch_length.setter
    def single_batch_length(self, value):
        self._single_batch_length = int(value)

    def process(self, workers, nmr_items, run_in_batches=None, single_batch_length=None):
        """Process all of the items using the callback function in the work packages.

        The idea is that a strategy can be chosen on the fly by for example testing the execution time of the callback
        functions. Alternatively, a strategy can be determined based on the available environments (in the WorkPackages)
        and/or by the total number of items to be processed.

        Args:
            workers (Worker): a list of workers
            nmr_items (int): an integer specifying the total number of items to be processed
            run_with_batches (boolean): a implementing class may overwrite run_in_batches with this parameter. If None
                the value is not used.
            single_batch_length (int): a implementing class may overwrite single_batch_length with this parameter.
                If None the value is not used.
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

    def _create_batches(self, range_start, range_end, run_in_batches=None, single_batch_length=None):
        """Created batches in the given range.

        If self.run_in_batches is False we will only return one batch covering the entire range. If self.run_in_batches
        is True we will create batches the size of self.single_batch_length.

        Args:
            range_start (int): the start of the range to create batches for
            range_end (int): the end of the range to create batches for
            run_with_batches (boolean): if other than None, use this as run_with_batches
            single_batch_length (int): if other than None, use this as single_batch_length

        Returns:
            list of list: list of batches which are (start, end) pairs
        """
        if run_in_batches is None:
            run_in_batches = self.run_in_batches

        if single_batch_length is None:
            single_batch_length = self.single_batch_length

        if run_in_batches:
            batches = []
            batch_pos = range_start

            while batch_pos < range_end:
                new_batch = (batch_pos, min(range_end, batch_pos + single_batch_length))
                batches.append(new_batch)
                batch_pos = new_batch[1]

            return batches
        return [(range_start, range_end)]

    def _run_batches(self, workers, batches):
        """Run a list of batches on each of the workers.

        This will enqueue on all the workers the batches in sequence and waits for completion of each batch before
        enqueueing the next one.

        Args:
            workers (list of Worker): the workers to use in the processing
            batches (list of lists): for each worker a list with the batches in format (start, end)
        """
        self._logger.debug('Preparing to run on {0} device(s)'.format(len(workers)))

        total_nmr_problems = 0
        most_nmr_batches = 0
        for workers_batches in batches:
            if len(workers_batches) > most_nmr_batches:
                most_nmr_batches = len(workers_batches)

            for batch in workers_batches:
                total_nmr_problems += batch[1] - batch[0]
        problems_seen = 0

        start_time = timeit.default_timer()
        for batch_nmr in range(most_nmr_batches):
            events = []
            for worker_ind, worker in enumerate(workers):
                if batch_nmr < len(batches[worker_ind]):
                    self._logger.debug('Going to run batch {0} on device {1} with range ({2}, {3})'.format(
                        batch_nmr, worker_ind, *batches[worker_ind][batch_nmr]))

                    events.append(self._try_processing(worker, *batches[worker_ind][batch_nmr]))
                    problems_seen += batches[worker_ind][batch_nmr][1] - batches[worker_ind][batch_nmr][0]
            for event in events:
                event.wait()

            run_time = timeit.default_timer() - start_time
            current_percentage = problems_seen / float(total_nmr_problems)
            remaining_time = (run_time / current_percentage) - run_time

            self._logger.info('Processing is at {0:.2%}, time spent: {1}, time left: {2} (h:m:s).'.format(
                current_percentage,
                time.strftime('%H:%M:%S', time.gmtime(run_time)),
                time.strftime('%H:%M:%S', time.gmtime(remaining_time))))

        self._logger.debug('Ran all batches.')

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

    @classmethod
    def get_pretty_name(cls):
        """The pretty name of this routine.

        This is used to create an object of the implementing class using a factory, and is used in the logging.

        Returns:
            str: the pretty name of this routine.
        """
        return cls.__name__

class MetaLoadBalanceStrategy(LoadBalanceStrategy):

    def __init__(self, lb_strategy):
        """ Create a load balance strategy that uses another strategy to do the actual computations.

        Args:
            lb_strategy (LoadBalanceStrategy): The load balance strategy this class uses.
        """
        super(MetaLoadBalanceStrategy, self).__init__()
        self._lb_strategy = lb_strategy or EvenDistribution()

    @property
    def run_in_batches(self):
        """ Returns the value for the load balance strategy this class uses. """
        return self._lb_strategy.run_in_batches

    @run_in_batches.setter
    def run_in_batches(self, value):
        """ Sets the value for the load balance strategy this class uses. """
        self._lb_strategy.run_in_batches = value

    @property
    def single_batch_length(self):
        """ Returns the value for the load balance strategy this class uses. """
        return self._lb_strategy.single_batch_length

    @single_batch_length.setter
    def single_batch_length(self, value):
        """ Sets the value for the load balance strategy this class uses. """
        self._lb_strategy.single_batch_length = value


class EvenDistribution(LoadBalanceStrategy):
    """Give each worker exactly 1/nth of the work. This does not do any feedback load balancing."""

    def process(self, workers, nmr_items, run_in_batches=None, single_batch_length=None):
        items_per_worker = int(round(nmr_items / float(len(workers))))
        batches = []
        current_pos = 0

        for worker_ind in range(len(workers)):
            if worker_ind == len(workers) - 1:
                batches.append(self._create_batches(current_pos, nmr_items,
                                                    run_in_batches=run_in_batches,
                                                    single_batch_length=single_batch_length))
            else:
                batches.append(self._create_batches(current_pos, current_pos + items_per_worker,
                                                    run_in_batches=run_in_batches,
                                                    single_batch_length=single_batch_length))
                current_pos += items_per_worker

        self._run_batches(workers, batches)

    def get_used_cl_environments(self, cl_environments):
        return cl_environments


class RuntimeLoadBalancing(LoadBalanceStrategy):

    def __init__(self, test_percentage=10, run_in_batches=True, single_batch_length=1e5):
        """Distribute the work by trying to minimize the time taken.

        Args:
            test_percentage (float): The percentage of items to use for the run time duration test
                (divided by number of devices)
        """
        super(RuntimeLoadBalancing, self).__init__(run_in_batches=run_in_batches,
                                                   single_batch_length=single_batch_length)
        self.test_percentage = test_percentage

    def process(self, workers, nmr_items, run_in_batches=None, single_batch_length=None):
        durations = []
        start = 0
        for worker in workers:
            end = start + int(math.floor(nmr_items * (self.test_percentage/len(workers)) / 100))
            durations.append(self._test_duration(worker, start, end))
            start = end

        total_d = sum(durations)
        nmr_items_left = nmr_items - start

        batches = []
        for i in range(len(workers)):
            if i == len(workers) - 1:
                batches.append(self._create_batches(start, nmr_items,
                                                    run_in_batches=run_in_batches,
                                                    single_batch_length=single_batch_length))
            else:
                items = int(math.floor(nmr_items_left * (1 - (durations[i] / total_d))))
                batches.append(self._create_batches(start, start + items,
                                                    run_in_batches=run_in_batches,
                                                    single_batch_length=single_batch_length))
                start += items

        self._run_batches(workers, batches)

    def _test_duration(self, worker, start, end):
        s = timeit.default_timer()
        self._run_batches([worker], [self._create_batches(start, end)])
        return timeit.default_timer() - s

    def get_used_cl_environments(self, cl_environments):
        return cl_environments


class PreferSingleDeviceType(MetaLoadBalanceStrategy):

    def __init__(self, lb_strategy=None, device_type=None):
        """This is a meta load balance strategy, it uses the given strategy and prefers the use of the indicated device.

        Args:
            lb_strategy (LoadBalanceStrategy): The strategy this class uses in the background.
            device_type (str or cl.device_type): either a cl device type or a string like ('gpu', 'cpu' or 'apu').
                This variable indicates the type of device we want to use.
        """
        super(PreferSingleDeviceType, self).__init__(lb_strategy)
        self._device_type = device_type or cl.device_type.CPU
        if isinstance(device_type, string_types):
            self._device_type = device_type_from_string(device_type)

    def process(self, workers, nmr_items, run_in_batches=None, single_batch_length=None):
        specific_workers = [worker for worker in workers if worker.cl_environment.device_type == self._device_type]

        if specific_workers:
            self._lb_strategy.process(specific_workers, nmr_items, run_in_batches=run_in_batches,
                                      single_batch_length=single_batch_length)
        else:
            self._lb_strategy.process(workers, nmr_items, run_in_batches=run_in_batches,
                                      single_batch_length=single_batch_length)

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


class PreferSpecificEnvironment(MetaLoadBalanceStrategy):

    def __init__(self, lb_strategy=None, environment_nmr=0):
        """This is a meta load balance strategy, it prefers the use of a specific CL environment.

        Use this only when you are sure how the list of CL devices will look like. For example in use with parallel
        optimization of multiple subjects with each on a specific device.

        Args:
            lb_strategy (LoadBalanceStrategy): The strategy this class uses in the background.
            environment_nmr (int): the specific environment to use in the list of CL environments
        """
        super(PreferSpecificEnvironment, self).__init__(lb_strategy)
        self.environment_nmr = environment_nmr

    def process(self, workers, nmr_items, run_in_batches=None, single_batch_length=None):
        self._lb_strategy.process(workers, nmr_items, run_in_batches=run_in_batches,
                                  single_batch_length=single_batch_length)

    def get_used_cl_environments(self, cl_environments):
        return [cl_environments[self.environment_nmr]]