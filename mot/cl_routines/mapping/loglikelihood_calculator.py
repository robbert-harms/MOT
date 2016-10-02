import pyopencl as cl
import numpy as np
from ...utils import ParameterCLCodeGenerator, get_float_type_def
from ...cl_routines.base import CLRoutine
from ...load_balance_strategies import Worker


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class LogLikelihoodCalculator(CLRoutine):

    def calculate(self, model, parameters_dict, evaluation_model=None):
        """Calculate and return the residuals.

        Args:
            model (AbstractModel): The model to calculate the residuals of.
            parameters_dict (dict): The parameters to use in the evaluation of the model
            evaluation_model (EvaluationModel): the evaluation model to use for the log likelihood. If not given
                we use the one defined in the model.

        Returns:
            Return per voxel the log likelihood.
        """
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        nmr_problems = model.get_nmr_problems()
        log_likelihoods = np.zeros((nmr_problems,), dtype=np_dtype, order='C')
        parameters = np.require(model.get_initial_parameters(parameters_dict),
                                np_dtype, requirements=['C', 'A', 'O'])

        workers = self._create_workers(
            lambda cl_environment: _LogLikelihoodCalculatorWorker(cl_environment, self.get_compile_flags_list(),
                                                                  model, parameters,
                                                                  log_likelihoods, evaluation_model))
        self.load_balancer.process(workers, model.get_nmr_problems())

        return log_likelihoods


class _LogLikelihoodCalculatorWorker(Worker):

    def __init__(self, cl_environment, compile_flags, model, parameters, log_likelihoods, evaluation_model):
        super(_LogLikelihoodCalculatorWorker, self).__init__(cl_environment)

        self._model = model
        self._double_precision = model.double_precision
        self._log_likelihoods = log_likelihoods
        self._parameters = parameters
        self._evaluation_model = evaluation_model

        self._var_data_dict = model.get_problems_var_data()
        self._protocol_data_dict = model.get_problems_protocol_data()
        self._model_data_dict = model.get_model_data()

        self._all_buffers, self._likelihoods_buffer = self._create_buffers()
        self._kernel = self._build_kernel(compile_flags)

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start
        event = self._kernel.run_kernel(self._cl_run_context.queue, (int(nmr_problems), ), None, *self._all_buffers,
                                        global_offset=(int(range_start),))
        return [self._enqueue_readout(self._likelihoods_buffer, self._log_likelihoods, range_start, range_end, [event])]

    def _create_buffers(self):
        constant_buffers = self._generate_constant_buffers(self._protocol_data_dict, self._model_data_dict)

        likelihoods_buffer = cl.Buffer(self._cl_run_context.context,
                                       cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                       hostbuf=self._log_likelihoods)

        params_buffer = cl.Buffer(self._cl_run_context.context,
                                  cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                  hostbuf=self._parameters)

        var_data_buffers = []
        for data in self._var_data_dict.values():
            var_data_buffers.append(cl.Buffer(self._cl_run_context.context,
                                              cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                              hostbuf=data.get_opencl_data()))

        all_buffers = [params_buffer, likelihoods_buffer]
        all_buffers.extend(var_data_buffers)
        all_buffers.extend(constant_buffers)

        return all_buffers, likelihoods_buffer

    def _get_kernel_source(self):
        cl_func = self._model.get_log_likelihood_function('getLogLikelihood', evaluation_model=self._evaluation_model)
        nmr_params = self._parameters.shape[1]

        param_code_gen = ParameterCLCodeGenerator(self._cl_environment.device, self._var_data_dict,
                                                  self._protocol_data_dict, self._model_data_dict)

        kernel_param_names = ['global mot_float_type* params', 'global mot_float_type* log_likelihoods']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())
        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += cl_func
        kernel_source += '''
            __kernel void run_kernel(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);
                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''

                    mot_float_type x[''' + str(nmr_params) + '''];
                    for(int i = 0; i < ''' + str(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                    }

                    log_likelihoods[gid] = getLogLikelihood(&data, x);
            }
        '''
        return kernel_source
