"""Supports hardware level load balancing over multiple CL enabled devices.

This load balancing consists of three players, :class:`~mot.lib.cl_environments.CLEnvironment`,
:class:`LoadBalanceStrategy` and :class:`~mot.cl_routines.base.CLRoutine`.
Every :class:`~mot.cl_routines.base.CLRoutine` (such as the Optimizers and Samplers) requires, in order to do
computations, a list of :class:`~mot.lib.cl_environments.CLEnvironment` and a :class:`LoadBalanceStrategy`
implementation. The :class:`~mot.lib.cl_environments.CLEnvironment` encapsulate all information needed to run
computations on its contained device. The :class:`LoadBalanceStrategy` chooses which environments (i.e. devices) to use
for the computations and how to use them. The load balancing itself is done by appointing subsets of
problems to specific devices.
"""
import math
import timeit
import warnings
import pyopencl as cl
from .utils import device_type_from_string


__author__ = 'Robbert Harms'
__date__ = "2014-06-23"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class LoadBalanceStrategy:
    """Basic interface of a load balancing strategy.

    Every load balancer has the option to run the calculations in batches. The advantage of batches is that it is
    interruptable and it may prevent memory errors since we run with smaller buffers. The disadvantage is that it
    may be slower due to constant waiting to load the new kernel and due to GPU thread starvation.
    """

    def process(self, workers, nmr_items, run_in_batches=None, single_batch_length=None):
        """Process all of the items using the callback function in the work packages.

        The idea is that a strategy can be chosen on the fly by for example testing the execution time of the callback
        functions. Alternatively, a strategy can be determined based on the available environments (in the WorkPackages)
        and/or by the total number of items to be processed.

        Args:
            workers (List[Worker]): a list of workers
            nmr_items (int): an integer specifying the total number of items to be processed
            run_in_batches (boolean): a implementing class may overwrite run_in_batches with this parameter. If None
                the value is not used.
            single_batch_length (int): a implementing class may overwrite single_batch_length with this parameter.
                If None the value is not used.
        """
        raise NotImplementedError()

    def get_used_cl_environments(self, cl_environments):
        """Get a subset of CL environments that this strategy plans on using.

        The strategy can decide on which workers to use based on the CL environment of the worker. To prevent the
        calling function from generating workers that will not be used by this strategy, the calling function can
        ask this function which CL environments it will use.

        Args:
            cl_environments (list): the CL environments we were planning on using and were planning on generating
                workers for

        Returns:
            list: A proper subset of the CL environments or all of them.
                This should reflect the list of Cl environment we will use in :meth:`process`.
        """
        raise NotImplementedError()


class Worker:

    def __init__(self, cl_environment):
        """Create a new worker.

        Workload strategies use workers to perform the calculations, in a distributed way determined by the strategy.
        All computed results should be stored internally by the worker.

        Args:
            cl_environment (CLEnvironment): The cl environment, can be used to determine the load
        """
        self._cl_environment = cl_environment
        self._cl_context = cl_environment.context
        self._cl_queue = cl_environment.queue

    @property
    def cl_environment(self):
        """Get the used CL environment.

        Returns:
            cl_environment (CLEnvironment): The cl environment to use for calculations.
        """
        return self._cl_environment

    @property
    def cl_queue(self):
        """Get the queue this worker is using for its GPU computations.

        The load balancing routine will use this queue to flush and finish the computations.

        Returns:
            pyopencl queues: the queue used by this worker
        """
        return self._cl_queue

    def calculate(self, range_start, range_end):
        """Calculate for this problem the given range.

        The results of the computations must be stored internally.

        Args:
            range_start (int): The start of the processing range
            range_end (int): The end of the processing range
        """
        raise NotImplementedError()


class SimpleLoadBalanceStrategy(LoadBalanceStrategy):

    def __init__(self, run_in_batches=True, single_batch_length=1e4):
        """An abstract class for quickly implementing load balancing strategies.

        Args:
            run_in_batches (boolean): If we want to run the load per worker in batches or in one large run.
            single_batch_length (float): The length of a single batch, only used if run_in_batches is set to True.
                This will create batches this size and run each of them one after the other.

        Attributes:
            run_in_batches (boolean); See above.
            single_batch_length (boolean); See above.
        """
        self._run_in_batches = run_in_batches
        self._single_batch_length = single_batch_length

    @property
    def run_in_batches(self):
        return self._run_in_batches

    @property
    def single_batch_length(self):
        return self._single_batch_length

    def process(self, workers, nmr_items, run_in_batches=None, single_batch_length=None):
        raise NotImplementedError()

    def get_used_cl_environments(self, cl_environments):
        raise NotImplementedError()

    def _create_batches(self, range_start, range_end, run_in_batches=None, single_batch_length=None):
        """Created batches in the given range.

        If self.run_in_batches is False we will only return one batch covering the entire range. If self.run_in_batches
        is True we will create batches the size of self.single_batch_length.

        Args:
            range_start (int): the start of the range to create batches for
            range_end (int): the end of the range to create batches for
            run_in_batches (boolean): if other than None, use this as run_with_batches
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
            for start_pos in range(int(range_start), int(range_end), int(single_batch_length)):
                batches.append((start_pos, int(min(start_pos + single_batch_length, range_end))))
            return batches

        return [(range_start, range_end)]

    def _run_batches(self, workers, batches):
        """Run a list of batches on each of the workers.

        This will enqueue on all the workers the batches in sequence and waits for completion of each batch before
        enqueueing the next one.

        Args:
            workers (List[Worker]): the workers to use in the processing
            batches (list of lists): for each worker a list with the batches in format (start, end)
        """
        total_nmr_problems = 0
        most_nmr_batches = 0
        for workers_batches in batches:
            if len(workers_batches) > most_nmr_batches:
                most_nmr_batches = len(workers_batches)

            for batch in workers_batches:
                total_nmr_problems += batch[1] - batch[0]
        problems_seen = 0

        for batch_nmr in range(most_nmr_batches):

            for worker_ind, worker in enumerate(workers):
                if batch_nmr < len(batches[worker_ind]):
                    worker.calculate(int(batches[worker_ind][batch_nmr][0]), int(batches[worker_ind][batch_nmr][1]))
                    problems_seen += batches[worker_ind][batch_nmr][1] - batches[worker_ind][batch_nmr][0]
                    worker.cl_queue.flush()

            for worker in workers:
                worker.cl_queue.finish()


class MetaLoadBalanceStrategy(SimpleLoadBalanceStrategy):

    def __init__(self, lb_strategy):
        """ Create a load balance strategy that uses another strategy to do the actual computations.

        Args:
            lb_strategy (SimpleLoadBalanceStrategy): The load balance strategy this class uses.
        """
        super().__init__()
        self._lb_strategy = lb_strategy or EvenDistribution()

    def process(self, workers, nmr_items, run_in_batches=None, single_batch_length=None):
        raise NotImplementedError()

    def get_used_cl_environments(self, cl_environments):
        raise NotImplementedError()

    @property
    def run_in_batches(self):
        """ Returns the value for the load balance strategy this class uses. """
        return self._lb_strategy.run_in_batches

    @property
    def single_batch_length(self):
        """ Returns the value for the load balance strategy this class uses. """
        return self._lb_strategy.single_batch_length


class EvenDistribution(SimpleLoadBalanceStrategy):
    """Give each worker exactly 1/nth of the work."""

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


class RuntimeLoadBalancing(SimpleLoadBalanceStrategy):

    def __init__(self, test_percentage=10, run_in_batches=True, single_batch_length=1e6):
        """Distribute the work by trying to minimize the runtime.

        This first runs a batch of a small size to estimate the runtime per devices. Afterwards the
        problem instances are distributed such to minimize the overall time.

        Args:
            test_percentage (float): The total percentage of items to use for the run time duration test
        """
        super().__init__(run_in_batches=run_in_batches,
                                                   single_batch_length=single_batch_length)
        self.test_percentage = test_percentage

    def process(self, workers, nmr_items, run_in_batches=None, single_batch_length=None):
        durations = []
        start = 0
        for worker in workers:
            end = start + int(math.floor(nmr_items * (self.test_percentage / len(workers)) / 100))
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
            lb_strategy (SimpleLoadBalanceStrategy): The strategy this class uses in the background.
            device_type (str or cl.device_type): either a cl device type or a string like ('gpu', 'cpu' or 'apu').
                This variable indicates the type of device we want to use.
        """
        super().__init__(lb_strategy)
        self._device_type = device_type or cl.device_type.CPU
        if isinstance(device_type, str):
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
            lb_strategy (SimpleLoadBalanceStrategy): The strategy this class uses in the background.
        """
        super().__init__(device_type='GPU', lb_strategy=lb_strategy)


class PreferCPU(PreferSingleDeviceType):

    def __init__(self, lb_strategy=None):
        """This is a meta load balance strategy, it uses the given strategy and prefers the use of CPU's.

        Args:
            lb_strategy (SimpleLoadBalanceStrategy): The strategy this class uses in the background.
        """
        super().__init__(device_type='CPU', lb_strategy=lb_strategy)


class PreferSpecificEnvironment(MetaLoadBalanceStrategy):

    def __init__(self, lb_strategy=None, environment_nmr=0):
        """This is a meta load balance strategy, it prefers the use of a specific CL environment.

        Use this only when you are sure how the list of CL devices will look like. For example in use with parallel
        optimization of multiple subjects with each on a specific device.

        Args:
            lb_strategy (SimpleLoadBalanceStrategy): The strategy this class uses in the background.
            environment_nmr (int): the specific environment to use in the list of CL environments
        """
        super().__init__(lb_strategy)
        self.environment_nmr = environment_nmr

    def process(self, workers, nmr_items, run_in_batches=None, single_batch_length=None):
        self._lb_strategy.process(workers, nmr_items, run_in_batches=run_in_batches,
                                  single_batch_length=single_batch_length)

    def get_used_cl_environments(self, cl_environments):
        return [cl_environments[self.environment_nmr]]
