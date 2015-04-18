import warnings
import pyopencl as cl
import numpy as np
from ...tools import get_cl_double_extension_definer, \
    get_read_only_cl_mem_flags, set_correct_cl_data_type, get_write_only_cl_mem_flags, ParameterCLCodeGenerator
from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import WorkerConstructor


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ResidualCalculator(AbstractCLRoutine):

    def __init__(self, cl_environments=None, load_balancer=None):
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
        cl_environments = self.load_balancer.get_used_cl_environments(self.cl_environments)
        parameters = set_correct_cl_data_type(model.get_initial_parameters(parameters))

        nmr_inst_per_problem = model.get_nmr_inst_per_problem()
        nmr_problems = model.get_nmr_problems()

        errors = np.asmatrix(np.zeros((nmr_problems, nmr_inst_per_problem)).astype(np.float64))

        problems_var_data_dict = set_correct_cl_data_type(model.get_problems_var_data())
        problems_prtcl_data_dict = set_correct_cl_data_type(model.get_problems_prtcl_data())
        problems_fixed_data_dict = set_correct_cl_data_type(model.get_problems_fixed_data())

        def run_transformer_cb(cl_environment, start, end, buffered_dicts):
            warnings.simplefilter("ignore")
            kernel_source = self._get_kernel_source(model.get_model_eval_function('evaluateModel'),
                                                    model.get_observation_return_function('getObservation'),
                                                    parameters.shape[1],
                                                    nmr_inst_per_problem, problems_var_data_dict,
                                                    problems_prtcl_data_dict, problems_fixed_data_dict,
                                                    cl_environment)
            kernel = cl.Program(cl_environment.context, kernel_source).build(' '.join(cl_environment.compile_flags))

            return self._run_calculator(parameters, problems_var_data_dict, errors, buffered_dicts[0],
                                        buffered_dicts[1], start, end, cl_environment, kernel)

        worker_constructor = WorkerConstructor()
        workers = worker_constructor.generate_workers(cl_environments, run_transformer_cb,
                                                      data_dicts_to_buffer=(problems_prtcl_data_dict,
                                                                            problems_fixed_data_dict))

        self.load_balancer.process(workers, nmr_problems)

        return errors

    def _run_calculator(self, parameters, var_data_dict, errors_host, prtcl_data_buffers, fixed_data_buffers,
                        start, end, cl_environment, kernel):

        write_only_flags = get_write_only_cl_mem_flags(cl_environment)
        read_only_flags = get_read_only_cl_mem_flags(cl_environment)
        nmr_problems = end - start
        queue = cl_environment.get_new_queue()

        errors_buf = cl.Buffer(cl_environment.context, write_only_flags, hostbuf=errors_host[start:end, :])

        data_buffers = [cl.Buffer(cl_environment.context, read_only_flags, hostbuf=parameters[start:end, :]),
                        errors_buf]
        for data in var_data_dict.values():
            if len(data.shape) < 2:
                data_buffers.append(cl.Buffer(cl_environment.context, read_only_flags, hostbuf=data[start:end]))
            else:
                data_buffers.append(cl.Buffer(cl_environment.context, read_only_flags, hostbuf=data[start:end, :]))
        data_buffers.extend(prtcl_data_buffers)
        data_buffers.extend(fixed_data_buffers)

        kernel.get_errors(queue, (int(nmr_problems), ), None, *data_buffers)
        event = cl.enqueue_copy(queue, errors_host[start:end, :], errors_buf, is_blocking=False)

        return queue, event

    def _get_kernel_source(self, cl_func, observation_func, nmr_params, nmr_inst_per_problem, var_data_dict,
                           prtcl_data_dict, model_data_dict, environment):

        param_code_gen = ParameterCLCodeGenerator(environment.device, var_data_dict, prtcl_data_dict, model_data_dict)

        kernel_param_names = ['global double* params', 'global double* errors']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + repr(nmr_inst_per_problem) + '''
        '''
        kernel_source += get_cl_double_extension_definer(environment.platform)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += observation_func
        kernel_source += cl_func
        kernel_source += '''
            __kernel void get_errors(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);
                    double x[''' + repr(nmr_params) + '''];
                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''

                    for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + repr(nmr_params) + ''' + i];
                    }

                    global double* result = errors + gid * NMR_INST_PER_PROBLEM;

                    for(int i = 0; i < NMR_INST_PER_PROBLEM; i++){
                        result[i] = getObservation(&data, i) - evaluateModel(&data, x, i);
                    }
            }
        '''
        return kernel_source