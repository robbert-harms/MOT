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


class CalculateModelEstimates(CLRoutine):

    def calculate(self, model, parameters):
        """Evaluate the model for every problem and every observation and return the estimates.

        This only evaluates the model at the given data points. It does not use the problem data to calculate
        objective values.

        Args:
            model (AbstractModel): The model to evaluate.
            parameters (ndarray): The parameters to use in the evaluation of the model

        Returns:
            ndarray: Return per problem instance the evaluation per data point.
        """
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        nmr_inst_per_problem = model.get_nmr_inst_per_problem()

        parameters = np.require(parameters, np_dtype, requirements=['C', 'A', 'O'])

        nmr_problems = parameters.shape[0]
        evaluations = np.zeros((nmr_problems, nmr_inst_per_problem), dtype=np_dtype, order='C')

        workers = self._create_workers(lambda cl_environment: _EvaluateModelWorker(
            cl_environment, self.get_compile_flags_list(model.double_precision), model, parameters, evaluations))
        self.load_balancer.process(workers, nmr_problems)

        return evaluations


class _EvaluateModelWorker(Worker):

    def __init__(self, cl_environment, compile_flags, model, parameters, evaluations):
        super(_EvaluateModelWorker, self).__init__(cl_environment)

        self._model = model
        self._data_info = self._model.get_kernel_data_info()
        self._double_precision = model.double_precision
        self._evaluations = evaluations
        self._parameters = parameters

        self._all_buffers, self._evaluations_buffer = self._create_buffers()
        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start
        self._kernel.get_estimates(self._cl_run_context.queue, (int(nmr_problems), ), None,
                                   *self._all_buffers, global_offset=(int(range_start),))
        self._enqueue_readout(self._evaluations_buffer, self._evaluations, range_start, range_end)

    def _create_buffers(self):
        evaluations_buffer = cl.Buffer(self._cl_run_context.context,
                                       cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                       hostbuf=self._evaluations)

        all_buffers = [cl.Buffer(self._cl_run_context.context,
                                 cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                 hostbuf=self._parameters),
                       evaluations_buffer]

        for data in self._data_info.get_data():
            all_buffers.append(cl.Buffer(self._cl_run_context.context,
                                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data))

        return all_buffers, evaluations_buffer

    def _get_kernel_source(self):
        eval_function_info = self._model.get_model_eval_function()
        param_modifier = self._model.get_pre_eval_parameter_modifier()

        nmr_params = self._parameters.shape[1]

        kernel_param_names = ['global mot_float_type* params', 'global mot_float_type* estimates']
        kernel_param_names.extend(self._data_info.get_kernel_parameters())

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + str(self._model.get_nmr_inst_per_problem()) + '''
        '''
        kernel_source += get_float_type_def(self._model.double_precision)
        kernel_source += self._data_info.get_kernel_data_struct()
        kernel_source += eval_function_info.get_function()
        kernel_source += param_modifier.get_function()
        kernel_source += '''
            __kernel void get_estimates(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    ulong gid = get_global_id(0);
                    ''' + self._data_info.get_kernel_data_struct_initialization('data') + '''

                    mot_float_type x[''' + str(nmr_params) + '''];
                    for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                    }

                    global mot_float_type* result = estimates + gid * NMR_INST_PER_PROBLEM;
                    
                    ''' + param_modifier.get_name() + '''((void*)&data, x);
                    
                    for(uint i = 0; i < NMR_INST_PER_PROBLEM; i++){
                        result[i] = ''' + eval_function_info.get_name() + '''((void*)&data, x, i);
                    }
            }
        '''
        return kernel_source
