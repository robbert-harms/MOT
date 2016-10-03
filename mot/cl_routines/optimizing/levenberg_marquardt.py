import os
from pkg_resources import resource_filename
import pyopencl as cl
from mot.utils import ParameterCLCodeGenerator, get_float_type_def
from .base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class LevenbergMarquardt(AbstractParallelOptimizer):

    default_patience = 250

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=True, patience=None,
                 optimizer_options=None, **kwargs):
        """Use the Levenberg-Marquardt method to calculate the optimimum.

        Args:
            patience (int):
                Used to set the maximum number of iterations to patience*(number_of_parameters+1)
        """
        patience = patience or self.default_patience
        super(LevenbergMarquardt, self).__init__(cl_environments, load_balancer, use_param_codec, patience=patience,
                                                 optimizer_options=optimizer_options, **kwargs)

    def _get_worker_generator(self, *args):
        return lambda cl_environment: LevenbergMarquardtWorker(cl_environment, *args)


class LevenbergMarquardtWorker(AbstractParallelOptimizerWorker):

    def __init__(self, *args, **kwargs):
        super(LevenbergMarquardtWorker, self).__init__(*args, **kwargs)

        if self._model.get_nmr_inst_per_problem() < self._nmr_params:
            raise ValueError('The number of instances per problem must be greater than the number of parameters')

    def _create_buffers(self):
        all_buffers, parameters_buffer, return_code_buffer = super(LevenbergMarquardtWorker, self)._create_buffers()

        fjac_items = self._nmr_params * self._model.get_nmr_inst_per_problem() * self._starting_points.shape[0]
        fjac_buffer_size = fjac_items * self._starting_points.dtype.itemsize
        fjac_buffer = cl.Buffer(self._cl_run_context.context, cl.mem_flags.READ_WRITE, size=fjac_buffer_size)
        all_buffers.append(fjac_buffer)

        return all_buffers, parameters_buffer, return_code_buffer

    def _get_kernel_source(self):
        """Overwrite the default kernel source generation.

        We need to add a few extra buffers to the kernel and the lmmin call. This makes the optimizer
        more robust when using datasets with a high number of observations per problem instance.

        If we inline these buffers the CL compiler may throw a 'out of resources' error since it can fail to find
        a contiguous block of memory large enough for any of these buffers. We circumvent that now by creating them
        in global memory space.
        """
        nmr_params = self._nmr_params
        param_code_gen = ParameterCLCodeGenerator(self._cl_environment.device,
                                                  self._var_data_dict,
                                                  self._protocol_data_dict,
                                                  self._model_data_dict)

        kernel_param_names = ['global mot_float_type* params',
                              'global char* return_codes']
        kernel_param_names.extend(param_code_gen.get_kernel_param_names())
        kernel_param_names.append('global mot_float_type* fjac_all')

        optimizer_call_args = 'x, (const void*) &data'

        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += str(param_code_gen.get_data_struct())

        if self._use_param_codec:
            param_codec = self._model.get_parameter_codec()
            decode_func = param_codec.get_cl_decode_function('decodeParameters')
            kernel_source += decode_func + "\n"

        kernel_source += self._get_optimizer_cl_code()
        kernel_source += '''
            __kernel void minimize(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    int gid = get_global_id(0);

                    mot_float_type x[''' + str(nmr_params) + '''];
                    for(int i = 0; i < ''' + str(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                    }

                    global mot_float_type* fjac = fjac_all + gid * ''' \
                         + str(self._nmr_params * self._model.get_nmr_inst_per_problem()) + ''';

                    ''' + param_code_gen.get_data_struct_init_assignment('data') + '''
                    return_codes[gid] = (char) ''' + self._get_optimizer_call_name() + '''(''' \
                         + optimizer_call_args + ''', fjac);

                    ''' + ('decodeParameters(x);' if self._use_param_codec else '') + '''

                    for(int i = 0; i < ''' + str(nmr_params) + '''; i++){
                        params[gid * ''' + str(nmr_params) + ''' + i] = x[i];
                    }
                }
        '''
        return kernel_source

    def _get_evaluate_function(self):
        """Get the CL code for the evaluation function. This is called from _get_optimizer_cl_code.

        Implementing optimizers can change this if desired.

        Returns:
            str: the evaluation function.
        """
        kernel_source = ''
        kernel_source += self._model.get_objective_list_function('calculateObjectiveList')
        if self._use_param_codec:
            kernel_source += '''
                void evaluate(mot_float_type* x, const void* data, mot_float_type* result){
                    mot_float_type x_model[''' + str(self._nmr_params) + '''];
                    for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        x_model[i] = x[i];
                    }
                    decodeParameters(x_model);
                    calculateObjectiveList((optimize_data*)data, x_model, result);
                }
            '''
        else:
            kernel_source += '''
                void evaluate(mot_float_type* x, const void* data, mot_float_type* result){
                    calculateObjectiveList((optimize_data*)data, x, result);
                }
            '''
        return kernel_source

    def _get_optimization_function(self):
        params = {'NMR_PARAMS': self._nmr_params,
                  'PATIENCE': self._parent_optimizer.patience,
                  'NMR_INST_PER_PROBLEM': self._model.get_nmr_inst_per_problem()}

        optimizer_options = self._optimizer_options or {}
        option_defaults = {'step_bound': 100.0, 'scale_diag': 1}
        option_converters = {'scale_diag': lambda val: int(bool(val))}

        for option, default in option_defaults.items():
            v = optimizer_options.get(option, default)
            if option in option_converters:
                v = option_converters[option](v)
            params.update({option.upper(): v})

        body = open(os.path.abspath(resource_filename('mot', 'data/opencl/lmmin.pcl')), 'r').read()
        if params:
            body = body % params
        return body

    def _get_optimizer_call_name(self):
        return 'lmmin'
