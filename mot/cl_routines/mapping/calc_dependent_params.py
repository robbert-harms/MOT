import pyopencl as cl
from ...utils import results_to_dict, ParameterCLCodeGenerator, get_float_type_def
from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import Worker
import numpy as np


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CalculateDependentParameters(AbstractCLRoutine):

    def __init__(self, cl_environments, load_balancer, double_precision=False):
        """CL code for calculating the dependent parameters.

        Some of the models may contain parameter dependencies. We would like to return the maps for these parameters
        as well as all the other maps. Since the dependencies are specified in CL, we have to recourse to CL to
        calculate these maps.

        Args:
            double_precision (boolean): if we will use the double (True) or single floating (False) type for the calculations
        """
        super(CalculateDependentParameters, self).__init__(cl_environments, load_balancer)
        self._double_precision = double_precision

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
        np_dtype = np.float32
        if self._double_precision:
            np_dtype = np.float64

        results_list = np.zeros(
            (estimated_parameters_list[0].shape[0], len(dependent_parameter_names)),
            dtype=np_dtype, order='C')

        estimated_parameters = np.dstack(estimated_parameters_list).flatten()

        workers = self._create_workers(_CDPWorker, [fixed_param_values, len(estimated_parameters_list),
                                       estimated_parameters, parameters_listing,
                                       dependent_parameter_names, results_list, self._double_precision])
        self.load_balancer.process(workers, estimated_parameters_list[0].shape[0])

        return results_to_dict(results_list, [n[1] for n in dependent_parameter_names])


class _CDPWorker(Worker):

    def __init__(self, cl_environment, var_data_dict, nmr_estimated_params, estimated_parameters,
                 parameters_listing, dependent_parameter_names, results_list, double_precision):
        super(_CDPWorker, self).__init__(cl_environment)

        self._var_data_dict = var_data_dict
        self._nmr_estimated_params = nmr_estimated_params
        self._parameters_listing = parameters_listing
        self._dependent_parameter_names = dependent_parameter_names
        self._results_list = results_list
        self._double_precision = double_precision

        self._estimated_parameters = estimated_parameters
        self._kernel = self._build_kernel()

    def calculate(self, range_start, range_end):
        write_only_flags = self._cl_environment.get_write_only_cl_mem_flags()
        read_only_flags = self._cl_environment.get_read_only_cl_mem_flags()
        nmr_problems = int(range_end - range_start)

        ep_start = range_start * self._nmr_estimated_params
        ep_end = range_end * self._nmr_estimated_params

        estimated_parameters_buf = cl.Buffer(
            self._cl_context.context, read_only_flags,
            hostbuf=self._estimated_parameters[ep_start:ep_end])

        results_buf = cl.Buffer(self._cl_context.context, write_only_flags,
                                hostbuf=self._results_list[range_start:range_end, :])

        data_buffers = [estimated_parameters_buf, results_buf]

        for data in self._var_data_dict.values():
            if len(data.shape) < 2:
                data_buffers.append(cl.Buffer(self._cl_context.context, read_only_flags,
                                              hostbuf=data[range_start:range_end]))
            else:
                data_buffers.append(cl.Buffer(self._cl_context.context, read_only_flags,
                                              hostbuf=data[range_start:range_end, :]))

        self._kernel.transform(self._cl_context.queue, (nmr_problems, ), None, *data_buffers)
        event = cl.enqueue_copy(self._cl_context.queue, self._results_list[range_start:range_end, :], results_buf,
                                is_blocking=False)
        return event

    def _get_kernel_source(self):
        dependent_parameter_names = [n[0] for n in self._dependent_parameter_names]

        parameter_write_out = ''
        for i, p in enumerate(dependent_parameter_names):
            parameter_write_out += 'results[gid * ' + str(len(dependent_parameter_names)) + \
                                   ' + ' + str(i) + '] = ' + p + ";\n"

        param_code_gen = ParameterCLCodeGenerator(self._cl_environment.device, self._var_data_dict, {}, {})
        kernel_param_names = ['global MOT_FLOAT_TYPE* params', 'global MOT_FLOAT_TYPE* results']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += '''
            __kernel void transform(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);

                    ''' + param_code_gen.get_data_struct_init_assignment('data_var') + '''
                    optimize_data* data = &data_var;

                    MOT_FLOAT_TYPE x[''' + str(self._nmr_estimated_params) + '''];
                    int i = 0;
                    for(i = 0; i < ''' + str(self._nmr_estimated_params) + '''; i++){
                        x[i] = params[gid * ''' + str(self._nmr_estimated_params) + ''' + i];
                    }
                    ''' + self._parameters_listing + '''
                    ''' + parameter_write_out + '''
            }
        '''
        return kernel_source