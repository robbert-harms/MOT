import warnings
import pyopencl as cl
import numpy as np
from ...utils import get_cl_double_extension_definer, \
    get_read_only_cl_mem_flags, set_correct_cl_data_type, get_write_only_cl_mem_flags, ParameterCLCodeGenerator
from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import Worker


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class EvaluateModel(AbstractCLRoutine):

    def __init__(self, cl_environments, load_balancer):
        """Evaluate the model at each problem instance for each data point."""
        super(EvaluateModel, self).__init__(cl_environments, load_balancer)

    def calculate(self, model, parameters):
        """Evaluate the model at each problem instance for each data point.

        Args:
            model (AbstractModel): The model to calculate the residuals of.
            parameters ndarray): The parameters to use in the evaluation of the model

        Returns:
            ndarray: Return per problem instance the evaluation per data point.
        """
        parameters = set_correct_cl_data_type(model.get_initial_parameters(parameters))

        nmr_problems = model.get_nmr_problems()
        nmr_inst_per_problem = model.get_nmr_inst_per_problem()

        evaluations = np.asmatrix(np.zeros((nmr_problems, nmr_inst_per_problem)).astype(np.float64,
                                                                                        order='C', copy=False))

        var_data_dict = set_correct_cl_data_type(model.get_problems_var_data())
        prtcl_data_dict = set_correct_cl_data_type(model.get_problems_prtcl_data())
        fixed_data_dict = set_correct_cl_data_type(model.get_problems_fixed_data())

        workers = self._create_workers(_EvaluateModelWorker, model, parameters, evaluations,
                                       var_data_dict, prtcl_data_dict, fixed_data_dict)
        self.load_balancer.process(workers, nmr_problems)

        return evaluations


class _EvaluateModelWorker(Worker):

    def __init__(self, cl_environment, model, parameters, evaluations, var_data_dict, prtcl_data_dict, fixed_data_dict):
        super(_EvaluateModelWorker, self).__init__(cl_environment)

        self._model = model
        self._parameters = parameters
        self._nmr_params = parameters.shape[1]
        self._evaluations = evaluations
        self._var_data_dict = var_data_dict
        self._prtcl_data_dict = prtcl_data_dict
        self._fixed_data_dict = fixed_data_dict

        self._constant_buffers = self._generate_constant_buffers(self._prtcl_data_dict, self._fixed_data_dict)
        self._kernel = self._build_kernel()

    def calculate(self, range_start, range_end):
        write_only_flags = get_write_only_cl_mem_flags(self._cl_environment)
        read_only_flags = get_read_only_cl_mem_flags(self._cl_environment)
        nmr_problems = range_end - range_start

        evals_buf = cl.Buffer(self._cl_environment.context, write_only_flags,
                              hostbuf=self._evaluations[range_start:range_end, :])

        data_buffers = [cl.Buffer(self._cl_environment.context, read_only_flags,
                                  hostbuf=self._parameters[range_start:range_end, :]),
                        evals_buf]
        for data in self._var_data_dict.values():
            if len(data.shape) < 2:
                data_buffers.append(cl.Buffer(self._cl_environment.context, read_only_flags,
                                              hostbuf=data[range_start:range_end]))
            else:
                data_buffers.append(cl.Buffer(self._cl_environment.context, read_only_flags,
                                              hostbuf=data[range_start:range_end, :]))
        data_buffers.extend(self._constant_buffers)

        self._kernel.get_errors(self._queue, (int(nmr_problems), ), None, *data_buffers)
        event = cl.enqueue_copy(self._queue, self._evaluations[range_start:range_end, :], evals_buf, is_blocking=False)

        return event

    def _get_kernel_source(self):
        param_code_gen = ParameterCLCodeGenerator(self._cl_environment.device,
                                                  self._var_data_dict, self._prtcl_data_dict, self._fixed_data_dict)

        kernel_param_names = ['global double* params', 'global double* evals']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + repr(self._model.get_nmr_inst_per_problem()) + '''
        '''
        kernel_source += get_cl_double_extension_definer(self._cl_environment.platform)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += self._model.get_model_eval_function('evaluateModel')
        kernel_source += '''
            __kernel void get_errors(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);
                    double x[''' + repr(self._nmr_params) + '''];
                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''

                    for(int i = 0; i < ''' + repr(self._nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + repr(self._nmr_params) + ''' + i];
                    }

                    global double* result = evals + gid * NMR_INST_PER_PROBLEM;

                    for(int i = 0; i < NMR_INST_PER_PROBLEM; i++){
                        result[i] = evaluateModel(&data, x, i);
                    }
            }
        '''
        return kernel_source