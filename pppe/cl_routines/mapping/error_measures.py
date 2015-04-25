import warnings
import pyopencl as cl
import numpy as np
from ...utils import get_cl_double_extension_definer, \
    get_read_only_cl_mem_flags, set_correct_cl_data_type, get_write_only_cl_mem_flags
from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import WorkerConstructor


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ErrorMeasures(AbstractCLRoutine):

    def __init__(self, cl_environments=None, load_balancer=None):
        """Given a set of raw errors per voxel, calculate some interesting measures."""
        super(ErrorMeasures, self).__init__(cl_environments, load_balancer)

    def calculate(self, errors):
        """Given a set of raw errors per voxel, calculate some interesting measures.

        Args:
            errors (ndarray): The list with errors per problem instance.

        Returns:
            A dictionary containing (for each voxel)
                - Errors.sum_of_squares: the sum of squares of the errors
                - Errors.log_sum_of_squares: the log of this sum of squares
                - Errors.mean_squared: the mean over the squares of errors
        """
        cl_environments = self.load_balancer.get_used_cl_environments(self.cl_environments)

        errors = set_correct_cl_data_type(errors)
        measures = np.asmatrix(np.zeros((errors.shape[0], 3))).astype(np.float64)

        def process_cb(cl_environment, start, end, buffered_dicts):
            return self._process(errors, measures, start, end, cl_environment)

        worker_constructor = WorkerConstructor()
        workers = worker_constructor.generate_workers(cl_environments, process_cb)

        self.load_balancer.process(workers, errors.shape[0])

        return {'Errors.sum_of_squares': measures[:, 0],
                'Errors.log_sum_of_squares': measures[:, 1],
                'Errors.mean_squared': measures[:, 2]}

    def _process(self, errors, measures, start, end, cl_environment):
        warnings.simplefilter("ignore")
        kernel_source = self._get_kernel_source(errors.shape[1], cl_environment)
        kernel = cl.Program(cl_environment.context, kernel_source).build(' '.join(cl_environment.compile_flags))

        write_only_flags = get_write_only_cl_mem_flags(cl_environment)
        read_only_flags = get_read_only_cl_mem_flags(cl_environment)
        nmr_problems = end - start
        queue = cl_environment.get_new_queue()

        errors_buf = cl.Buffer(cl_environment.context, read_only_flags, hostbuf=errors[start:end, :])
        measures_buf = cl.Buffer(cl_environment.context, write_only_flags, hostbuf=measures[start:end, :])

        kernel.get_measures(queue, (int(nmr_problems), ), None, errors_buf, measures_buf)
        event = cl.enqueue_copy(queue, measures[start:end, :], measures_buf, is_blocking=False)
        return queue, event

    def _get_kernel_source(self, nmr_inst_per_problem, cl_environment):
        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + repr(nmr_inst_per_problem) + '''
        '''
        kernel_source += get_cl_double_extension_definer(cl_environment.platform)
        kernel_source += '''
            __kernel void get_measures(
                global double* errors,
                global double* measures
                ){
                    int gid = get_global_id(0);

                    double sumsq = 0.0;
                    for(int i = 0; i < NMR_INST_PER_PROBLEM; i++){
                        sumsq += pown(errors[gid * NMR_INST_PER_PROBLEM + i], 2);
                    }

                    measures[gid * 3] = sumsq;
                    measures[gid * 3 + 1] = log(sumsq);
                    measures[gid * 3 + 2] = sumsq/NMR_INST_PER_PROBLEM;
            }
        '''
        return kernel_source