import pyopencl as cl
import numpy as np
from ...utils import ParameterCLCodeGenerator, get_float_type_def
from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import Worker


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ResidualCalculator(AbstractCLRoutine):

    def __init__(self, cl_environments, load_balancer):
        """Calculate the residuals, that is the errors, per problem instance per data point."""
        super(ResidualCalculator, self).__init__(cl_environments, load_balancer)

    def calculate(self, model, parameters_dict):
        """Calculate and return the residuals.

        Args:
            model (AbstractModel): The model to calculate the residuals of.
            parameters_dict (dict): The parameters to use in the evaluation of the model

        Returns:
            Return per voxel the errors (eval - data) per protocol item
        """
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        nmr_inst_per_problem = model.get_nmr_inst_per_problem()
        nmr_problems = model.get_nmr_problems()

        residuals = np.asmatrix(np.zeros((nmr_problems, nmr_inst_per_problem), dtype=np_dtype, order='C'))

        parameters = model.get_initial_parameters(parameters_dict)

        workers = self._create_workers(lambda cl_environment: _ResidualCalculatorWorker(cl_environment, model,
                                                                                        parameters, residuals))
        self.load_balancer.process(workers, model.get_nmr_problems())

        return residuals


class _ResidualCalculatorWorker(Worker):

    def __init__(self, cl_environment, model, parameters, residuals):
        super(_ResidualCalculatorWorker, self).__init__(cl_environment)

        self._model = model
        self._double_precision = model.double_precision
        self._residuals = residuals
        self._parameters = parameters

        self._var_data_dict = model.get_problems_var_data()
        self._protocol_data_dict = model.get_problems_protocol_data()
        self._model_data_dict = model.get_model_data()

        self._constant_buffers = self._generate_constant_buffers(self._protocol_data_dict, self._model_data_dict)

        self._kernel = self._build_kernel()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        all_buffers, errors_buffer = self._create_buffers(range_start, range_end)

        self._kernel.get_errors(self._cl_run_context.queue, (int(nmr_problems), ), None, *all_buffers)
        event = cl.enqueue_copy(self._cl_run_context.queue, self._residuals[range_start:range_end, :], errors_buffer,
                                is_blocking=False)

        return event

    def _create_buffers(self, range_start, range_end):
        write_only_flags = self._cl_environment.get_write_only_cl_mem_flags()
        read_only_flags = self._cl_environment.get_read_only_cl_mem_flags()

        errors_buffer = cl.Buffer(self._cl_run_context.context,
                                  write_only_flags, hostbuf=self._residuals[range_start:range_end, :])

        all_buffers = [cl.Buffer(self._cl_run_context.context,
                                 read_only_flags, hostbuf=self._parameters[range_start:range_end, :]),
                       errors_buffer]

        for data in self._var_data_dict.values():
            all_buffers.append(cl.Buffer(self._cl_run_context.context,
                                         read_only_flags, hostbuf=data.get_opencl_data()[range_start:range_end, ...]))

        all_buffers.extend(self._constant_buffers)
        return all_buffers, errors_buffer

    def _get_kernel_source(self):
        cl_func = self._model.get_model_eval_function('evaluateModel')
        nmr_inst_per_problem = self._model.get_nmr_inst_per_problem()
        nmr_params = self._parameters.shape[1]
        observation_func = self._model.get_observation_return_function('getObservation')
        param_code_gen = ParameterCLCodeGenerator(self._cl_environment.device, self._var_data_dict,
                                                  self._protocol_data_dict, self._model_data_dict)

        kernel_param_names = ['global mot_float_type* params', 'global mot_float_type* errors']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + str(nmr_inst_per_problem) + '''
        '''

        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += observation_func
        kernel_source += cl_func
        kernel_source += '''
            __kernel void get_errors(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);
                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''

                    mot_float_type x[''' + str(nmr_params) + '''];
                    for(int i = 0; i < ''' + str(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                    }

                    global mot_float_type* result = errors + gid * NMR_INST_PER_PROBLEM;

                    for(int i = 0; i < NMR_INST_PER_PROBLEM; i++){
                        result[i] = getObservation(&data, i) - evaluateModel(&data, x, i);
                    }
            }
        '''
        return kernel_source
