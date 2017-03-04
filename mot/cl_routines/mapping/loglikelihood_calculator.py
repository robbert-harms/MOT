from collections import Mapping

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


class LogLikelihoodCalculator(CLRoutine):

    def calculate(self, model, parameters, evaluation_model=None):
        """Calculate and return the log likelihood of the given model under the given parameters.

        Args:
            model (AbstractModel): The model to calculate the full log likelihood for.
            parameters (dict or ndarray): The parameters to use in the evaluation of the model
                If a dict is given we assume it is with values for a set of parameters
                If an ndarray is given we assume that we have data for all parameters.
            evaluation_model (EvaluationModel): the evaluation model to use for the log likelihood. If not given
                we use the one defined in the model.

        Returns:
            Return per voxel the log likelihood.
        """
        parameters = self._initialize_parameters(parameters, model)
        log_likelihoods = self._initialize_result_array(model)

        workers = self._create_workers(
            lambda cl_environment: _LogLikelihoodCalculatorWorker(cl_environment,
                                                                  self.get_compile_flags_list(model.double_precision),
                                                                  model, parameters,
                                                                  log_likelihoods, evaluation_model))
        self.load_balancer.process(workers, model.get_nmr_problems())

        return log_likelihoods

    def _initialize_parameters(self, parameters, model):
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        if isinstance(parameters, Mapping):
            return np.require(model.get_initial_parameters(parameters), np_dtype, requirements=['C', 'A', 'O'])

        return np.require(parameters, np_dtype, requirements=['C', 'A', 'O'])

    def _initialize_result_array(self, model):
        nmr_problems = model.get_nmr_problems()
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64
        return np.zeros((nmr_problems,), dtype=np_dtype, order='C')


class _LogLikelihoodCalculatorWorker(Worker):

    def __init__(self, cl_environment, compile_flags, model, parameters, log_likelihoods, evaluation_model):
        super(_LogLikelihoodCalculatorWorker, self).__init__(cl_environment)

        self._model = model
        self._double_precision = model.double_precision
        self._log_likelihoods = log_likelihoods
        self._parameters = parameters
        self._evaluation_model = evaluation_model

        self._all_buffers, self._likelihoods_buffer = self._create_buffers()
        self._kernel = self._build_kernel(compile_flags)

    def __del__(self):
        for buffer in self._all_buffers:
            buffer.release()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start
        event = self._kernel.run_kernel(self._cl_run_context.queue, (int(nmr_problems), ), None, *self._all_buffers,
                                        global_offset=(int(range_start),))
        return [self._enqueue_readout(self._likelihoods_buffer, self._log_likelihoods, range_start, range_end, [event])]

    def _create_buffers(self):
        likelihoods_buffer = cl.Buffer(self._cl_run_context.context,
                                       cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                       hostbuf=self._log_likelihoods)

        params_buffer = cl.Buffer(self._cl_run_context.context,
                                  cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                  hostbuf=self._parameters)

        all_buffers = [params_buffer, likelihoods_buffer]

        for data in self._model.get_data():
            all_buffers.append(cl.Buffer(self._cl_run_context.context,
                                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data))

        return all_buffers, likelihoods_buffer

    def _get_kernel_source(self):
        cl_func = self._model.get_log_likelihood_function('getLogLikelihood', evaluation_model=self._evaluation_model)
        nmr_params = self._parameters.shape[1]

        kernel_param_names = ['global mot_float_type* params', 'global mot_float_type* log_likelihoods']
        kernel_param_names.extend(self._model.get_kernel_param_names(self._cl_environment.device))
        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += self._model.get_kernel_data_struct(self._cl_environment.device)
        kernel_source += cl_func
        kernel_source += '''
            __kernel void run_kernel(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);
                    ''' + self._model.get_kernel_data_struct_initialization(self._cl_environment.device, 'data') + '''

                    mot_float_type x[''' + str(nmr_params) + '''];
                    for(int i = 0; i < ''' + str(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                    }

                    log_likelihoods[gid] = getLogLikelihood((void*)&data, x);
            }
        '''
        return kernel_source
