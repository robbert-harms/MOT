import os
from pkg_resources import resource_filename
import pyopencl as cl
from mot.utils import get_float_type_def
from .base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class LevenbergMarquardt(AbstractParallelOptimizer):

    default_patience = 250

    def __init__(self, patience=None, step_bound=None, scale_diag=None, optimizer_settings=None, **kwargs):
        """Use the Levenberg-Marquardt method to calculate the optimimum.

        Args:
            patience (int): Used to set the maximum number of iterations to patience*(number_of_parameters+1)
        """
        patience = patience or self.default_patience

        optimizer_settings = optimizer_settings or {}

        keyword_values = {}
        keyword_values['step_bound'] = step_bound
        keyword_values['scale_diag'] = scale_diag

        option_defaults = {'step_bound': 100.0, 'scale_diag': 1}

        def get_value(option_name):
            value = keyword_values.get(option_name)
            if value is None:
                value = optimizer_settings.get(option_name)
            if value is None:
                value = option_defaults[option_name]
            return value

        for option in option_defaults:
            optimizer_settings.update({option: get_value(option)})

        super(LevenbergMarquardt, self).__init__(patience=patience, optimizer_settings=optimizer_settings, **kwargs)

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

        kernel_param_names = ['global mot_float_type* restrict params',
                              'global char* restrict return_codes']
        kernel_param_names.extend(self._data_struct_manager.get_kernel_arguments())
        kernel_param_names.append('global mot_float_type* restrict fjac_all')

        optimizer_call_args = 'x, (void*)&data'

        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += self._data_struct_manager.get_struct_definition()

        kernel_source += self._get_optimizer_cl_code()
        kernel_source += '''
            __kernel void minimize(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    ulong gid = get_global_id(0);

                    mot_float_type x[''' + str(nmr_params) + '''];
                    for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                    }

                    global mot_float_type* fjac = fjac_all + gid * ''' \
                         + str(self._nmr_params * self._model.get_nmr_inst_per_problem()) + ''';

                    mot_data_struct data = ''' + self._data_struct_manager.get_struct_init_string('gid') + ''';
                    return_codes[gid] = (char) ''' + self._get_optimizer_call_name() + '''(''' \
                         + optimizer_call_args + ''', fjac);

                    for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
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
        objective_func = self._model.get_objective_per_observation_function()
        param_modifier = self._model.get_pre_eval_parameter_modifier()

        kernel_source = ''
        kernel_source += objective_func.get_cl_code()
        kernel_source += param_modifier.get_cl_code()
        kernel_source += '''
            void evaluate(mot_float_type* x, void* data_void, mot_float_type* result){
                mot_data_struct* data = (mot_data_struct*)data_void;
                
                mot_float_type x_model[''' + str(self._model.get_nmr_estimable_parameters()) + '''];
                for(uint i = 0; i < ''' + str(self._model.get_nmr_estimable_parameters()) + '''; i++){
                    x_model[i] = x[i];
                }
                ''' + param_modifier.get_cl_function_name() + '''(data, x_model);
                
                for(uint i = 0; i < ''' + str(self._model.get_nmr_inst_per_problem()) + '''; i++){
                    // the model expects the L1 norm, while the LM method takes the L2 norm. Taking square root here.
                    result[i] = sqrt(fabs(''' + objective_func.get_cl_function_name() + '''(data, x_model, i)));
                }
            }
        '''
        return kernel_source

    def _get_optimization_function(self):
        params = {'NMR_PARAMS': self._nmr_params,
                  'PATIENCE': self._parent_optimizer.patience,
                  'NMR_INST_PER_PROBLEM': self._model.get_nmr_inst_per_problem(),
                  'USER_TOL_MULT': 30}

        optimizer_settings = self._optimizer_settings or {}
        option_defaults = {'step_bound': 100.0, 'scale_diag': 1, 'usertol_mult': 30}
        option_converters = {'scale_diag': lambda val: int(bool(val))}

        for option, default in option_defaults.items():
            v = optimizer_settings.get(option, default)
            if option in option_converters:
                v = option_converters[option](v)
            params.update({option.upper(): v})

        body = open(os.path.abspath(resource_filename('mot', 'data/opencl/lmmin.cl')), 'r').read()
        if params:
            body = body % params
        return body

    def _get_optimizer_call_name(self):
        return 'lmmin'
