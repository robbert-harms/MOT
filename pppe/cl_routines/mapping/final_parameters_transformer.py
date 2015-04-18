import warnings
import pyopencl as cl
from ...tools import get_cl_double_extension_definer, \
    get_read_write_cl_mem_flags, get_read_only_cl_mem_flags, set_correct_cl_data_type, ParameterCLCodeGenerator
from ...cl_routines.base import AbstractCLRoutine
from ...load_balance_strategies import WorkerConstructor


__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class FinalParametersTransformer(AbstractCLRoutine):

    def __init__(self, cl_environments=None, load_balancer=None):
        """CL code for applying the final parameter transforms.

        Some of the models may contain parameter dependencies. These dependencies may have side-effects and change
        multiple parameters. This change occurs right before the parameters are entered in the model evaluation function
        right before the actual model is evaluated.

        Suppose an optimization routine finds a set of parameters X to the the optimal set of parameters. In the
        evaluation function this set of parameters might have been transformed to a new set of parameters X' by the
        parameter dependencies. Since we, in the end, are interested in the set of parameters X', we have to apply
        the exact same transformations at the end of the optimization routine as happened in the evaluation function.

        This class supports running those transformations.
        """
        super(FinalParametersTransformer, self).__init__(cl_environments, load_balancer)

    def transform(self, model, parameters):
        """This transforms the parameters matrix in place. Using the final parameters transforms."""
        cl_environments = self.load_balancer.get_used_cl_environments(self.cl_environments)

        if model.get_final_parameter_transformations():
            parameters = set_correct_cl_data_type(parameters)
            var_data_dict = set_correct_cl_data_type(model.get_problems_var_data())
            prtcl_data_dict = set_correct_cl_data_type(model.get_problems_prtcl_data())
            fixed_data_dict = set_correct_cl_data_type(model.get_problems_fixed_data())

            def run_transformer_cb(cl_environment, start, end, buffered_dicts):
                kernel_source = self._get_kernel_source(
                    model.get_final_parameter_transformations('applyFinalParameterTransformations'),
                    parameters.shape[1],
                    var_data_dict,
                    prtcl_data_dict,
                    fixed_data_dict,
                    cl_environment)
                warnings.simplefilter("ignore")
                kernel = cl.Program(cl_environment.context, kernel_source).build(' '.join(cl_environment.compile_flags))
                return self._run_transformer(parameters, var_data_dict, buffered_dicts[0], buffered_dicts[1],
                                             start, end, cl_environment, kernel)

            worker_constructor = WorkerConstructor()
            workers = worker_constructor.generate_workers(cl_environments, run_transformer_cb,
                                                          data_dicts_to_buffer=(prtcl_data_dict, fixed_data_dict))

            self.load_balancer.process(workers, model.get_nmr_problems())

        return parameters

    def _run_transformer(self, parameters, var_data_dict, prtcl_data_buffers, fixed_data_buffers,
                         start, end, cl_environment, kernel):
        read_write_flags = get_read_write_cl_mem_flags(cl_environment)
        read_only_flags = get_read_only_cl_mem_flags(cl_environment)
        nmr_problems = end - start
        queue = cl_environment.get_new_queue()

        data_buffers = []
        parameters_buf = cl.Buffer(cl_environment.context, read_write_flags, hostbuf=parameters[start:end, :])
        data_buffers.append(parameters_buf)
        for data in var_data_dict.values():
            if len(data.shape) < 2:
                data_buffers.append(cl.Buffer(cl_environment.context, read_only_flags, hostbuf=data[start:end]))
            else:
                data_buffers.append(cl.Buffer(cl_environment.context, read_only_flags, hostbuf=data[start:end, :]))
        data_buffers.extend(prtcl_data_buffers)
        data_buffers.extend(fixed_data_buffers)

        kernel.transform(queue, (int(nmr_problems), ), None, *data_buffers)
        event = cl.enqueue_copy(queue, parameters[start:end, :], parameters_buf, is_blocking=False)
        return queue, event

    def _get_kernel_source(self, cl_final_param_transforms, nmr_params, var_data_dict, prtcl_data_dict,
                           model_data_dict, environment):
        param_code_gen = ParameterCLCodeGenerator(environment.device, var_data_dict, prtcl_data_dict, model_data_dict)

        kernel_param_names = ['global double* params']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = get_cl_double_extension_definer(environment.platform)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += cl_final_param_transforms
        kernel_source += '''
            __kernel void transform(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);
                    double x[''' + repr(nmr_params) + '''];
                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''

                    for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + repr(nmr_params) + ''' + i];
                    }

                    applyFinalParameterTransformations(&data, x);

                    for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                        params[gid * ''' + repr(nmr_params) + ''' + i] = x[i];
                    }
            }
        '''
        return kernel_source