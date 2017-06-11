import pyopencl as cl
import numpy as np
from ...utils import get_float_type_def
from ...cl_routines.base import CLRoutine
from ...load_balance_strategies import Worker


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ResidualCalculator(CLRoutine):

    def __init__(self, cl_environments=None, load_balancer=None):
        """Calculate the residuals, that is the errors, per problem instance per data point."""
        super(ResidualCalculator, self).__init__(cl_environments=cl_environments, load_balancer=load_balancer)

    def calculate(self, model, parameters_dict, model_estimates=None):
        """Calculate and return the residuals.

        Args:
            model (AbstractModel): The model to calculate the residuals of.
            parameters_dict (dict): The parameters to use in the evaluation of the model
            model_estimates (ndarray): The model estimates of the model if available. If not given we
                calculate the model estimates using the initial parameters from the model.

        Returns:
            Return per voxel the errors (eval - data) per protocol item
        """
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        nmr_inst_per_problem = model.get_nmr_inst_per_problem()
        nmr_problems = model.get_nmr_problems()

        residuals = np.zeros((nmr_problems, nmr_inst_per_problem), dtype=np_dtype, order='C')
        parameters = np.require(model.get_initial_parameters(parameters_dict), np_dtype, requirements=['C', 'A', 'O'])

        if model_estimates is not None:
            model_estimates = np.require(model_estimates, np_dtype, requirements=['C', 'A', 'O'])

        workers = self._create_workers(lambda cl_environment: _ResidualCalculatorWorker(
            cl_environment, self.get_compile_flags_list(model.double_precision), model, parameters,
            residuals, model_estimates))
        self.load_balancer.process(workers, model.get_nmr_problems())

        return residuals


class _ResidualCalculatorWorker(Worker):

    def __init__(self, cl_environment, compile_flags, model, parameters, residuals, model_estimates=None):
        super(_ResidualCalculatorWorker, self).__init__(cl_environment)

        self._model = model
        self._double_precision = model.double_precision
        self._residuals = residuals
        self._parameters = parameters
        self._model_estimates = model_estimates

        self._all_buffers, self._residuals_buffer = self._create_buffers()
        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

    def __del__(self):
        for buffer in self._all_buffers:
            buffer.release()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start
        self._kernel.get_errors(self._cl_run_context.queue, (int(nmr_problems), ), None, *self._all_buffers,
                                global_offset=(int(range_start),))
        self._enqueue_readout(self._residuals_buffer, self._residuals, range_start, range_end)

    def _create_buffers(self):
        errors_buffer = cl.Buffer(self._cl_run_context.context,
                                  cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                  hostbuf=self._residuals)

        if self._model_estimates is None:
            all_buffers = [cl.Buffer(self._cl_run_context.context,
                                     cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                     hostbuf=self._parameters),
                           errors_buffer]
        else:
            all_buffers = [cl.Buffer(self._cl_run_context.context,
                                     cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                     hostbuf=self._model_estimates),
                           errors_buffer]

        for data in self._model.get_data():
            all_buffers.append(cl.Buffer(self._cl_run_context.context,
                                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data))

        return all_buffers, errors_buffer

    def _get_kernel_source(self):
        nmr_inst_per_problem = self._model.get_nmr_inst_per_problem()
        nmr_params = self._parameters.shape[1]

        if self._model_estimates is None:
            kernel_param_names = ['global mot_float_type* params', 'global mot_float_type* errors']
        else:
            kernel_param_names = ['global mot_float_type* model_estimates', 'global mot_float_type* errors']

        kernel_param_names.extend(self._model.get_kernel_param_names(self._cl_environment.device))

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + str(nmr_inst_per_problem) + '''
        '''

        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += self._model.get_kernel_data_struct(self._cl_environment.device)
        kernel_source += self._model.get_observation_return_function('getObservation')
        if self._model_estimates is None:
            kernel_source += self._model.get_model_eval_function('evaluateModel')
            kernel_source += '''
                __kernel void get_errors(
                    ''' + ",\n".join(kernel_param_names) + '''
                    ){
                        ulong gid = get_global_id(0);
                        ''' + self._model.get_kernel_data_struct_initialization(self._cl_environment.device, 'data') + '''

                        mot_float_type x[''' + str(nmr_params) + '''];
                        for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                            x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                        }

                        global mot_float_type* result = errors + gid * NMR_INST_PER_PROBLEM;

                        for(uint i = 0; i < NMR_INST_PER_PROBLEM; i++){
                            result[i] = getObservation((void*)&data, i) - evaluateModel((void*)&data, x, i);
                        }
                }
            '''
        else:
            kernel_source += '''
                __kernel void get_errors(
                    ''' + ",\n".join(kernel_param_names) + '''
                    ){
                        ulong gid = get_global_id(0);
                        ''' + self._model.get_kernel_data_struct_initialization(self._cl_environment.device,
                                                                                'data') + '''

                        global mot_float_type* result = errors + gid * NMR_INST_PER_PROBLEM;

                        for(uint i = 0; i < NMR_INST_PER_PROBLEM; i++){
                            result[i] = getObservation(&data, i) - model_estimates[i + gid * NMR_INST_PER_PROBLEM];
                        }
                }
            '''
        return kernel_source
