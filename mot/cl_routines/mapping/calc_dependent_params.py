import pyopencl as cl
from ...utils import results_to_dict, get_float_type_def
from ...cl_routines.base import CLRoutine
from ...load_balance_strategies import Worker
import numpy as np


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CalculateDependentParameters(CLRoutine):

    def __init__(self, double_precision=False, **kwargs):
        """CL code for calculating the dependent parameters.

        Some of the models may contain parameter dependencies. We would like to return the maps for these parameters
        as well as all the other maps. Since the dependencies are specified in CL, we have to recourse to CL to
        calculate these maps.

        Args:
            double_precision (boolean): if we will use the double (True) or single floating (False) type
                for the calculations
        """
        super(CalculateDependentParameters, self).__init__(**kwargs)
        self._double_precision = double_precision

    def calculate(self, model, estimated_parameters_list, parameters_listing, dependent_parameter_names):
        """Calculate the dependent parameters

        This uses the calculated parameters in the results dictionary to run the parameters_listing in CL to obtain
        the maps for the dependent parameters.

        Args:
            model (mot.model_interfaces.OptimizeModelInterface): the model for which to get the dependent parameters
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

        estimated_parameters = np.require(np.dstack(estimated_parameters_list),
                                          np_dtype, requirements=['C', 'A', 'O'])[0, ...]

        workers = self._create_workers(
            lambda cl_environment: _CDPWorker(cl_environment, self.get_compile_flags_list(self._double_precision),
                                              model, len(estimated_parameters_list),
                                              estimated_parameters, parameters_listing,
                                              dependent_parameter_names, results_list, self._double_precision))
        self.load_balancer.process(workers, estimated_parameters_list[0].shape[0])

        return results_to_dict(results_list, [n[1] for n in dependent_parameter_names])


class _CDPWorker(Worker):

    def __init__(self, cl_environment, compile_flags, model, nmr_estimated_params, estimated_parameters,
                 parameters_listing, dependent_parameter_names, results_list, double_precision):
        super(_CDPWorker, self).__init__(cl_environment)

        self._nmr_estimated_params = nmr_estimated_params
        self._parameters_listing = parameters_listing
        self._dependent_parameter_names = dependent_parameter_names
        self._results_list = results_list
        self._double_precision = double_precision

        self._model = model

        self._estimated_parameters = estimated_parameters
        self._all_buffers, self._results_list_buffer = self._create_buffers()
        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

    def __del__(self):
        for buffer in self._all_buffers:
            buffer.release()

    def calculate(self, range_start, range_end):
        nmr_problems = int(range_end - range_start)

        self._kernel.transform(self._cl_run_context.queue, (int(nmr_problems), ), None, *self._all_buffers,
                               global_offset=(int(range_start),))
        self._enqueue_readout(self._results_list_buffer, self._results_list, range_start, range_end)

    def _create_buffers(self):
        estimated_parameters_buf = cl.Buffer(self._cl_run_context.context,
                                             cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                             hostbuf=self._estimated_parameters)

        results_buffer = cl.Buffer(self._cl_run_context.context,
                                   cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                   hostbuf=self._results_list)

        data_buffers = [estimated_parameters_buf, results_buffer]

        for data in self._model.get_data():
            data_buffers.append(cl.Buffer(self._cl_run_context.context,
                                          cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data))

        return data_buffers, results_buffer

    def _get_kernel_source(self):
        dependent_parameter_names = [n[0] for n in self._dependent_parameter_names]

        parameter_write_out = ''
        for i, p in enumerate(dependent_parameter_names):
            parameter_write_out += 'results[gid * ' + str(len(dependent_parameter_names)) + \
                                   ' + ' + str(i) + '] = ' + p + ";\n"

        kernel_param_names = ['global mot_float_type* params', 'global mot_float_type* results']
        kernel_param_names.extend(self._model.get_kernel_param_names(self._cl_environment.device))

        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += self._model.get_kernel_data_struct(self._cl_environment.device)
        kernel_source += '''
            __kernel void transform(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    ulong gid = get_global_id(0);

                    ''' + self._model.get_kernel_data_struct_initialization(self._cl_environment.device,
                                                                            'data_var') + '''
                    ''' + self._model.get_kernel_data_struct_type() + '''* data = &data_var;

                    mot_float_type x[''' + str(self._nmr_estimated_params) + '''];

                    for(uint i = 0; i < ''' + str(self._nmr_estimated_params) + '''; i++){
                        x[i] = params[gid * ''' + str(self._nmr_estimated_params) + ''' + i];
                    }
                    ''' + self._parameters_listing + '''
                    ''' + parameter_write_out + '''
            }
        '''
        return kernel_source
