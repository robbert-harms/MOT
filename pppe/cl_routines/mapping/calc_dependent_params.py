import pyopencl as cl
from ...utils import get_cl_double_extension_definer, \
    get_read_only_cl_mem_flags, set_correct_cl_data_type, \
    get_write_only_cl_mem_flags, results_to_dict, ParameterCLCodeGenerator
from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import PreferCPU, Worker
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

    def calculate(self, fixed_param_values, estimated_parameters_list, parameters_listing, dependent_parameter_names):
        """Calculate the dependent parameters

        This uses the calculated parameters in the results dictionary to run the parameters_listing in CL to obtain
        the maps for the dependent parameters.

        Args:
            fixed_param_values (dict): The dictionary with fixed parameter values to be used for parameters that are
                constant.
            estimated_parameters_list (list of ndarray): The list with the one-dimensional
                ndarray of estimated parameters
            parameters_listing (str): The parameters listing in CL
            dependent_parameter_names (list of list of str): Per parameter we would like to obtain the CL name and the
                result map name. For example: (('Wball_w', 'Wball.w'),)

        Returns:
            dict: A dictionary with the calculated maps for the dependent parameters.
        """
        fixed_param_values = set_correct_cl_data_type(fixed_param_values)

        results_list = np.zeros(
            (estimated_parameters_list[0].shape[0], len(dependent_parameter_names)),
            dtype=np.float64, order='C')

        estimated_parameters = set_correct_cl_data_type(np.dstack(estimated_parameters_list).ravel().reshape((-1, 1)))

        workers = self._create_workers(_CDPWorker, fixed_param_values, estimated_parameters_list, estimated_parameters,
                                       parameters_listing, dependent_parameter_names, results_list)
        self.load_balancer.process(workers, estimated_parameters_list[0].shape[0])

        return results_to_dict(results_list, [n[1] for n in dependent_parameter_names])


class _CDPWorker(Worker):

    def __init__(self, cl_environment, fixed_param_values, estimated_parameters_list, estimated_parameters,
                 parameters_listing, dependent_parameter_names, results_list):
        super(_CDPWorker, self).__init__(cl_environment)

        self._fixed_param_values = fixed_param_values
        self._estimated_parameters_list = estimated_parameters_list
        self._parameters_listing = parameters_listing
        self._dependent_parameter_names = dependent_parameter_names
        self._results_list = results_list

        self._constant_buffers = self._generate_constant_buffers(self._fixed_param_values)
        self._estimated_parameters = estimated_parameters
        self._kernel = self._build_kernel()

    def calculate(self, range_start, range_end):
        write_only_flags = get_write_only_cl_mem_flags(self._cl_environment)
        read_only_flags = get_read_only_cl_mem_flags(self._cl_environment)
        nmr_problems = range_end - range_start

        estimated_parameters_buf = cl.Buffer(self._cl_environment.context, read_only_flags,
                                             hostbuf=self._estimated_parameters[range_start:range_end, :])
        results_buf = cl.Buffer(self._cl_environment.context, write_only_flags,
                                hostbuf=self._results_list[range_start:range_end, :])

        data_buffers = [estimated_parameters_buf, results_buf]
        data_buffers.extend(self._constant_buffers)

        self._kernel.transform(self._queue, (int(nmr_problems), ), None, *data_buffers)
        event = cl.enqueue_copy(self._queue, self._results_list[range_start:range_end, :], results_buf, is_blocking=False)
        return event

    def _get_kernel_source(self):
        dependent_parameter_names = [n[0] for n in self._dependent_parameter_names]

        parameter_write_out = ''
        for i, p in enumerate(dependent_parameter_names):
            parameter_write_out += 'results[gid * ' + repr(len(dependent_parameter_names)) + \
                                   ' + ' + repr(i) + '] = ' + p + ";\n"

        param_code_gen = ParameterCLCodeGenerator(self._cl_environment.device, self._fixed_param_values, {}, {})
        kernel_param_names = ['global double* params', 'global double* results']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = get_cl_double_extension_definer(self._cl_environment.platform)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += '''
            __kernel void transform(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);

                    ''' + param_code_gen.get_data_struct_init_assignment('data_var') + '''
                    optimize_data* data = &data_var;

                    double x[''' + repr(len(self._estimated_parameters_list)) + '''];
                    int i = 0;
                    for(i = 0; i < ''' + repr(len(self._estimated_parameters_list)) + '''; i++){
                        x[i] = params[gid * ''' + repr(len(self._estimated_parameters_list)) + ''' + i];
                    }
                    ''' + self._parameters_listing + '''
                    ''' + parameter_write_out + '''
            }
        '''
        return kernel_source