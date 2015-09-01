import pyopencl as cl
import numpy as np
from ...utils import get_cl_pragma_double, ParameterCLCodeGenerator, get_float_type_def
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

    def calculate(self, model, parameters):
        """Calculate and return the residuals.

        Args:
            model (AbstractModel): The model to calculate the residuals of.
            parameters (ndarray): The parameters to use in the evaluation of the model

        Returns:
            Return per voxel the errors (eval - data) per scheme line
        """
        np_dtype = np.float32
        if model.use_double:
            np_dtype = np.float64

        nmr_inst_per_problem = model.get_nmr_inst_per_problem()
        nmr_problems = model.get_nmr_problems()

        residuals = np.asmatrix(np.zeros((nmr_problems, nmr_inst_per_problem)).astype(np_dtype, copy=False))

        parameters = model.get_initial_parameters(parameters)
        parameters = parameters.astype(np_dtype, order='C', copy=False)

        workers = self._create_workers(_ResidualCalculatorWorker, model, parameters, residuals)
        self.load_balancer.process(workers, model.get_nmr_problems())

        return residuals


class _ResidualCalculatorWorker(Worker):

    def __init__(self, cl_environment, model, parameters, residuals):
        super(_ResidualCalculatorWorker, self).__init__(cl_environment)

        self._model = model
        self._use_double = model.use_double
        self._residuals = residuals
        self._parameters = parameters

        self._var_data_dict = model.get_problems_var_data()
        self._prtcl_data_dict = model.get_problems_prtcl_data()
        self._fixed_data_dict = model.get_problems_fixed_data()

        self._constant_buffers = self._generate_constant_buffers(self._prtcl_data_dict, self._fixed_data_dict)

        self._kernel = self._build_kernel()

    def calculate(self, range_start, range_end):
        write_only_flags = self._cl_environment.get_write_only_cl_mem_flags()
        read_only_flags = self._cl_environment.get_read_only_cl_mem_flags()
        nmr_problems = range_end - range_start

        errors_buf = cl.Buffer(self._cl_environment.context,
                               write_only_flags, hostbuf=self._residuals[range_start:range_end, :])

        data_buffers = [cl.Buffer(self._cl_environment.context,
                                  read_only_flags, hostbuf=self._parameters[range_start:range_end, :]),
                        errors_buf]

        for data in self._var_data_dict.values():
            if len(data.shape) < 2:
                data_buffers.append(cl.Buffer(self._cl_environment.context,
                                              read_only_flags, hostbuf=data[range_start:range_end]))
            else:
                data_buffers.append(cl.Buffer(self._cl_environment.context,
                                              read_only_flags, hostbuf=data[range_start:range_end, :]))

        data_buffers.extend(self._constant_buffers)

        self._kernel.get_errors(self._queue, (int(nmr_problems), ), None, *data_buffers)
        event = cl.enqueue_copy(self._queue, self._residuals[range_start:range_end, :], errors_buf, is_blocking=False)

        return event

    def _get_kernel_source(self):
        cl_func = self._model.get_model_eval_function('evaluateModel')
        nmr_inst_per_problem = self._model.get_nmr_inst_per_problem()
        nmr_params = self._parameters.shape[1]
        observation_func = self._model.get_observation_return_function('getObservation')
        param_code_gen = ParameterCLCodeGenerator(self._cl_environment.device, self._var_data_dict,
                                                  self._prtcl_data_dict, self._fixed_data_dict)

        kernel_param_names = ['global model_float* params', 'global model_float* errors']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + str(nmr_inst_per_problem) + '''
        '''
        kernel_source += get_cl_pragma_double()
        kernel_source += get_float_type_def(self._use_double)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += observation_func
        kernel_source += cl_func
        kernel_source += '''
            __kernel void get_errors(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);
                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''

                    double x[''' + str(nmr_params) + '''];
                    for(int i = 0; i < ''' + str(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                    }

                    global model_float* result = errors + gid * NMR_INST_PER_PROBLEM;

                    for(int i = 0; i < NMR_INST_PER_PROBLEM; i++){
                        result[i] = getObservation(&data, i) - evaluateModel(&data, x, i);
                    }
            }
        '''
        return kernel_source
