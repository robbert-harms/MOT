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


class FinalParametersTransformer(AbstractCLRoutine):

    def __init__(self, cl_environments, load_balancer):
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
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        parameters = parameters.astype(np_dtype, order='C', copy=False)
        var_data_dict = model.get_problems_var_data()
        prtcl_data_dict = model.get_problems_prtcl_data()
        fixed_data_dict = model.get_problems_fixed_data()

        if model.get_final_parameter_transformations():
            workers = self._create_workers(_FPTWorker, [model, parameters, var_data_dict,
                                           prtcl_data_dict, fixed_data_dict])
            self.load_balancer.process(workers, model.get_nmr_problems())

        return parameters


class _FPTWorker(Worker):

    def __init__(self, cl_environment, model, parameters, var_data_dict, prtcl_data_dict, fixed_data_dict):
        super(_FPTWorker, self).__init__(cl_environment)

        self._parameters = parameters
        self._nmr_params = parameters.shape[1]
        self._model = model
        self._var_data_dict = var_data_dict
        self._prtcl_data_dict = prtcl_data_dict
        self._fixed_data_dict = fixed_data_dict
        self._double_precision = model.double_precision
        self._constant_buffers = self._generate_constant_buffers(self._prtcl_data_dict, self._fixed_data_dict)
        self._kernel = self._build_kernel()

    def calculate(self, range_start, range_end):
        read_write_flags = self._cl_environment.get_read_write_cl_mem_flags()
        read_only_flags = self._cl_environment.get_read_only_cl_mem_flags()
        nmr_problems = range_end - range_start

        data_buffers = []
        parameters_buf = cl.Buffer(self._cl_run_context.context, read_write_flags,
                                   hostbuf=self._parameters[range_start:range_end, :])
        data_buffers.append(parameters_buf)
        for data in self._var_data_dict.values():
            if len(data.shape) < 2:
                data_buffers.append(cl.Buffer(self._cl_run_context.context, read_only_flags,
                                              hostbuf=data[range_start:range_end]))
            else:
                data_buffers.append(cl.Buffer(self._cl_run_context.context, read_only_flags,
                                              hostbuf=data[range_start:range_end, :]))
        data_buffers.extend(self._constant_buffers)

        self._kernel.transform(self._cl_run_context.queue, (int(nmr_problems), ), None, *data_buffers)
        event = cl.enqueue_copy(self._cl_run_context.queue, self._parameters[range_start:range_end, :],
                                parameters_buf, is_blocking=False)
        return event

    def _get_kernel_source(self):
        param_code_gen = ParameterCLCodeGenerator(self._cl_environment.device,
                                                  self._var_data_dict, self._prtcl_data_dict, self._fixed_data_dict)

        kernel_param_names = ['global MOT_FLOAT_TYPE* params']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())

        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += param_code_gen.get_data_struct()
        kernel_source += self._model.get_final_parameter_transformations('applyFinalParameterTransformations')
        kernel_source += '''
            __kernel void transform(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);
                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''

                    MOT_FLOAT_TYPE x[''' + str(self._nmr_params) + '''];
                    for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(self._nmr_params) + ''' + i];
                    }

                    applyFinalParameterTransformations(&data, x);

                    for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        params[gid * ''' + str(self._nmr_params) + ''' + i] = x[i];
                    }
            }
        '''
        return kernel_source