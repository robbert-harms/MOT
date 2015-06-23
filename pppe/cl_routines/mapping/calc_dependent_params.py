import warnings
import pyopencl as cl
from ...utils import get_cl_double_extension_definer, \
    get_read_only_cl_mem_flags, set_correct_cl_data_type, \
    get_write_only_cl_mem_flags, results_to_dict
from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import WorkerConstructor, PreferCPU
import numpy as np


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CalculateDependentParameters(AbstractCLRoutine):

    def __init__(self, cl_environments=None, load_balancer=None):
        """CL code for calculating the dependent parameters.

        Some of the models may contain parameter dependencies. We would like to return the maps for these parameters
        as well as all the other maps. Since the dependencies are specified in CL, we have to recourse to CL to
        calculate these maps.
        """
        if not load_balancer:
            load_balancer = PreferCPU()
        super(CalculateDependentParameters, self).__init__(cl_environments, load_balancer)

    def calculate(self, estimated_parameters_list, parameters_listing, dependent_parameter_names):
        """Calculate the dependent parameters

        This uses the calculated parameters in the results dictionary to run the parameters_listing in CL to obtain
        the maps for the dependent parameters.

        Args:
            estimated_parameters_list (list of ndarray): The list with the one-dimensional
                ndarray of estimated parameters
            parameters_listing (str): The parameters listing in CL
            dependent_parameter_names (list of list of str): Per parameter we would like to obtain the CL name and the
                result map name. For example: (('Wball_w', 'Wball.w'),)

        Returns:
            dict: A dictionary with the calculated maps for the dependent parameters.
        """
        cl_environments = self.load_balancer.get_used_cl_environments(self.cl_environments)

        estimated_parameters = set_correct_cl_data_type(np.dstack(estimated_parameters_list).ravel().reshape((-1, 1)))
        results_list = np.zeros(
            (estimated_parameters_list[0].shape[0], len(dependent_parameter_names)),
            dtype=np.float64, order='C')

        def run_transformer_cb(cl_environment, start, end, buffered_dicts):
            kernel_source = self._get_kernel_source(
                parameters_listing,
                len(estimated_parameters_list),
                [n[0] for n in dependent_parameter_names],
                cl_environment)

            warnings.simplefilter("ignore")
            kernel = cl.Program(cl_environment.context, kernel_source).build(' '.join(cl_environment.compile_flags))
            return self._run_cl(estimated_parameters, results_list, start, end, cl_environment, kernel)

        worker_constructor = WorkerConstructor()
        workers = worker_constructor.generate_workers(cl_environments, run_transformer_cb)

        self.load_balancer.process(workers, estimated_parameters_list[0].shape[0])

        return results_to_dict(results_list, [n[1] for n in dependent_parameter_names])

    def _run_cl(self, estimated_parameters, results_list, start, end, cl_environment, kernel):
        write_only_flags = get_write_only_cl_mem_flags(cl_environment)
        read_only_flags = get_read_only_cl_mem_flags(cl_environment)
        nmr_problems = end - start
        queue = cl_environment.get_new_queue()

        estimated_parameters_buf = cl.Buffer(cl_environment.context, read_only_flags,
                                             hostbuf=estimated_parameters[start:end, :])
        results_buf = cl.Buffer(cl_environment.context, write_only_flags,
                                hostbuf=results_list[start:end, :])

        data_buffers = [estimated_parameters_buf, results_buf]

        kernel.transform(queue, (int(nmr_problems), ), None, *data_buffers)
        event = cl.enqueue_copy(queue, results_list[start:end, :], results_buf, is_blocking=False)
        return queue, event

    def _get_kernel_source(self, parameters_listing, nmr_params, dependent_parameter_names, environment):
        parameter_write_out = ''
        for i, p in enumerate(dependent_parameter_names):
            parameter_write_out += 'results[gid * ' + repr(len(dependent_parameter_names)) + \
                                   ' + ' + repr(i) + '] = ' + p + ";\n"

        kernel_param_names = ['global double* params', 'global double* results']
        kernel_source = get_cl_double_extension_definer(environment.platform)
        kernel_source += '''
            __kernel void transform(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);
                    double x[''' + repr(nmr_params) + '''];
                    int i = 0;
                    for(i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + repr(nmr_params) + ''' + i];
                    }
                    ''' + parameters_listing + '''
                    ''' + parameter_write_out + '''
            }
        '''
        return kernel_source