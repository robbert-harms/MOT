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


class ObjectiveCalculator(CLRoutine):

    def calculate(self, model, parameters):
        """Calculate and return the objective function of the given model for the given parameters.

        This evaluates the model and compares it to the problem data to get objective values. This returns
        the objective value per problem instance, for an objective function value per observation per problem use
        the :class:`~.objective_list_calculator.ObjectiveListCalculator`.

        Args:
            model (AbstractModel): The model to calculate the objective function of.
            parameters (ndarray): The parameters to use for calculating the objective values.

        Returns:
            ndarray: Returns per voxel the objective function value
        """
        parameters = self._initialize_parameters(parameters, model)
        objective_values = self._initialize_result_array(model)

        workers = self._create_workers(
            lambda cl_environment: _ObjectiveCalculatorWorker(
                cl_environment, self.get_compile_flags_list(model.double_precision), model,
                parameters, objective_values))
        self.load_balancer.process(workers, model.get_nmr_problems())

        return objective_values

    def _initialize_parameters(self, parameters, model):
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64
        return np.require(parameters, np_dtype, requirements=['C', 'A', 'O'])

    def _initialize_result_array(self, model):
        nmr_problems = model.get_nmr_problems()
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64
        return np.zeros((nmr_problems,), dtype=np_dtype, order='C')


class _ObjectiveCalculatorWorker(Worker):

    def __init__(self, cl_environment, compile_flags, model, parameters, objective_values):
        super(_ObjectiveCalculatorWorker, self).__init__(cl_environment)

        self._model = model
        self._data_info = self._model.get_kernel_data_info()
        self._double_precision = model.double_precision
        self._objective_values = objective_values
        self._parameters = parameters

        self._all_buffers, self._objective_values_buffer = self._create_buffers()
        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start
        self._kernel.run_kernel(self._cl_run_context.queue, (int(nmr_problems), ), None, *self._all_buffers,
                                global_offset=(int(range_start),))
        self._enqueue_readout(self._objective_values_buffer, self._objective_values, range_start, range_end)

    def _create_buffers(self):
        objective_value_buffer = cl.Buffer(self._cl_run_context.context,
                                           cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                           hostbuf=self._objective_values)

        params_buffer = cl.Buffer(self._cl_run_context.context,
                                  cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                  hostbuf=self._parameters)

        all_buffers = [params_buffer, objective_value_buffer]

        for data in self._data_info.get_data():
            all_buffers.append(cl.Buffer(self._cl_run_context.context,
                                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data))

        return all_buffers, objective_value_buffer

    def _get_kernel_source(self):
        objective_function = self._model.get_objective_per_observation_function()
        param_modifier = self._model.get_pre_eval_parameter_modifier()

        nmr_params = self._parameters.shape[1]

        kernel_param_names = ['global mot_float_type* params', 'global mot_float_type* objective_values']
        kernel_param_names.extend(self._data_info.get_kernel_parameters())
        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += self._data_info.get_kernel_data_struct()
        kernel_source += objective_function.get_function()
        kernel_source += param_modifier.get_function()
        kernel_source += '''
            double _evaluate(const void* data, mot_float_type* x){
                ''' + param_modifier.get_name() + '''((void*)&data, x);
                
                double sum = 0;
                for(uint i = 0; i < ''' + str(self._model.get_nmr_inst_per_problem()) + '''; i++){
                    sum += pown(''' + objective_function.get_name() + '''(data, x, i), 2);
                }
                return sum;
            }
        '''
        kernel_source += '''
            __kernel void run_kernel(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    ulong gid = get_global_id(0);
                    ''' + self._data_info.get_kernel_data_struct_initialization('data') + '''

                    mot_float_type x[''' + str(nmr_params) + '''];
                    for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                    }

                    objective_values[gid] = _evaluate((void*)&data, x);
            }
        '''
        return kernel_source
