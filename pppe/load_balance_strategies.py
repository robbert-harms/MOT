import math
import time
import numpy as np
import pyopencl as cl
from .tools import get_read_only_cl_mem_flags


__author__ = 'Robbert Harms'
__date__ = "2014-06-23"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Worker(object):

    def __init__(self, cl_environment, cb_function):
        """A work package consists of an CL environment and a callback function.

        The callback function is a (pre-initialized) function that accepts a start and end point of data to process. The
        full signature of this (python) function is as follows:
            queue, event = cb_function(start, end)

        The idea is that the workload strategy can execute this function in a way it seems fit for the strategy. During
        determining the strategy, all items computed should be stored internally by the callback function.

        Args:
            cl_environment (CLEnvironment): The cl environment to use for calculations.
            cb_function (python function handler): The callback function with the signature cb_function(start, end)
        """
        self._cl_environment = cl_environment
        self._cb_function = cb_function

    @property
    def cl_environment(self):
        """Get the used CL environment.

        Returns:
            cl_environment (CLEnvironment): The cl environment to use for calculations.
        """
        return self._cl_environment

    @property
    def cb_function(self):
        """The callback function used by this worker.

        Returns:
            cb_function (python function handler): The callback function
        """
        return self._cb_function


class WorkerConstructor(object):

    def generate_workers(self, cl_environments, callback_generator, data_dicts_to_buffer=None):
        """Generate workers using the given information.

        Args:
            cl_environments (list): A list with the CL environments for which to create the work packages

            wp_compute_callback (python function): The function that generate the compute callback function
                that is used in the work package.
                Signature:
                    def wp_minimizer_generator(cl_environment, prtcl_data_buffers, fixed_data_buffers, start, end):
                        return python_cb function

            prtcl_data_dict (dict): The dictionary with constant data, can be empty
            fixed_data_dict (dict): The dictionary with the fixd data, can be empty

        Returns:
            The workers (instances of LoadBalanceStrategy.Worker) that the load balancer
            may use to calculate the objectives.
        """
        workers = []
        for cl_environment in cl_environments:
            buffered = []
            if data_dicts_to_buffer:
                for e in data_dicts_to_buffer:
                    buffered.append(self._generate_buffers(cl_environment, e))

            def cb_function_gen(cl_environment=cl_environment, buffered=buffered):
                def compute_cb(start, end):
                    return callback_generator(cl_environment, start, end, buffered)
                return compute_cb

            workers.append(Worker(cl_environment, cb_function_gen()))

        return workers

    def _generate_buffers(self, cl_environment, data_dict):
        result = []
        if data_dict:
            for data in data_dict.values():
                if isinstance(data, np.ndarray):
                    result.append(cl.Buffer(cl_environment.context, get_read_only_cl_mem_flags(cl_environment),
                                            hostbuf=data))
                else:
                    result.append(data)
        return result


class LoadBalanceStrategy(object):

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


class EvenDistribution(LoadBalanceStrategy):
    """Distribute the work evenly over all the available work packages."""

    def process(self, workers, nmr_items):
        queues = []
        finish_events = []
        items_per_worker = round(nmr_items / float(len(workers)))
        current_pos = 0
        for i in range(len(workers)):
            if i == len(workers) - 1:
                queue, event = workers[i].cb_function(current_pos, nmr_items)
            else:
                queue, event = workers[i].cb_function(current_pos, current_pos + items_per_worker)
                current_pos += items_per_worker

            queues.append(queue)
            finish_events.append(event)

        for finish_event in finish_events:
            if finish_event:
                finish_event.wait()

    def get_used_cl_environments(self, cl_environments):
        return cl_environments


class RuntimeLoadBalancing(LoadBalanceStrategy):

    def __init__(self, test_percentage=10):
        """Distribute the work by trying to minimize the time taken.

        Args:
            test_percentage (float): The percentage of items to use for the run time duration test
                (divided by number of devices)
        """
        self.test_percentage = test_percentage

    def process(self, workers, nmr_items):
        durations = []
        start = 0
        for worker in workers:
            end = start + int(math.floor(nmr_items * (self.test_percentage/len(workers)) / 100))
            durations.append(self.test_duration(worker, start, end))
            start = end

        total_d = sum(durations)
        nmr_items_left = nmr_items - start

        queues = []
        finish_events = []
        for i in range(len(workers)):
            if i == len(workers) - 1:
                queue, event = workers[i].cb_function(start, nmr_items)
            else:
                items = int(math.floor(nmr_items_left * (1 - (durations[i] / total_d))))
                queue, event = workers[i].cb_function(start, start + items)
                start += items

            queues.append(queue)
            finish_events.append(event)

        for finish_event in finish_events:
            if finish_event:
                finish_event.wait()

    def test_duration(self, worker, start, end):
        s = time.time()
        queue, event = worker.cb_function(start, end)
        if event:
            event.wait()
        return time.time() - s

    def get_used_cl_environments(self, cl_environments):
        return cl_environments


class PreferGPU(LoadBalanceStrategy):

    def __init__(self, lb_strategy=EvenDistribution()):
        """This is a meta load balance strategy, it uses the given strategy and prefers the use of GPU's.

        Args:
            lb_strategy (LoadBalanceStrategy): The strategy this class uses in the background.
        """
        self._lb_strategy = lb_strategy

    def process(self, workers, nmr_items):
        gpu_list = [wp for wp in workers if wp.cl_environment.is_gpu]

        if gpu_list:
            self._lb_strategy.process(gpu_list, nmr_items)
        else:
            self._lb_strategy.process(workers, nmr_items)

    def get_used_cl_environments(self, cl_environments):
        gpu_list = [cl_environment for cl_environment in cl_environments if cl_environment.is_gpu]

        if gpu_list:
            return gpu_list
        else:
            return cl_environments


class PreferCPU(LoadBalanceStrategy):

    def __init__(self, lb_strategy=EvenDistribution()):
        """This is a meta load balance strategy, it uses the given strategy and prefers the use of CPU's.

        Args:
            lb_strategy (LoadBalanceStrategy): The strategy this class uses in the background.
        """
        self._lb_strategy = lb_strategy

    def process(self, workers, nmr_items):
        cpu_list = [wp for wp in workers if wp.cl_environment.is_cpu]

        if cpu_list:
            self._lb_strategy.process(cpu_list, nmr_items)
        else:
            self._lb_strategy.process(workers, nmr_items)

    def get_used_cl_environments(self, cl_environments):
        cpu_list = [cl_environment for cl_environment in cl_environments if cl_environment.is_cpu]

        if cpu_list:
            return cpu_list
        else:
            return cl_environments