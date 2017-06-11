from collections import Mapping

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


class ObjectiveListCalculator(CLRoutine):

    def __init__(self, **kwargs):
        """Calculate the objective list, that is it can compute the get_objective_list_function of the given model.

        This evaluates the model and compares it to the problem data to get objective values. As such it will return
        a value per problem instance per data point. For an objective function summarized over the observations use
        the :class:`~.objective_calculator.ObjectiveCalculator`.
        """
        super(ObjectiveListCalculator, self).__init__(**kwargs)

    def calculate(self, model, parameters):
        """Calculate and return the objective lists.

        Args:
            model (AbstractModel): The model to calculate the residuals of.
            parameters (dict or ndarray): The parameters to use in the evaluation of the model
                If a dict is given we assume it is with values for a set of parameters
                If an ndarray is given we assume that we have data for all parameters.

        Returns:
            Return per voxel the objective value (application of the function: "noise_model(eval - data)")
                per protocol item.
        """
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        nmr_inst_per_problem = model.get_nmr_inst_per_problem()
        nmr_problems = model.get_nmr_problems()

        if isinstance(parameters, Mapping):
            parameters = np.require(model.get_initial_parameters(parameters), np_dtype,
                                    requirements=['C', 'A', 'O'])
        else:
            parameters = np.require(parameters, np_dtype, requirements=['C', 'A', 'O'])

        objectives = np.zeros((nmr_problems, nmr_inst_per_problem), dtype=np_dtype, order='C')

        workers = self._create_workers(lambda cl_environment: _ObjectiveListCalculatorWorker(
            cl_environment, self.get_compile_flags_list(model.double_precision), model, parameters, objectives))
        self.load_balancer.process(workers, model.get_nmr_problems())

        return objectives


class _ObjectiveListCalculatorWorker(Worker):

    def __init__(self, cl_environment, compile_flags, model, parameters, objectives):
        super(_ObjectiveListCalculatorWorker, self).__init__(cl_environment)

        self._model = model
        self._double_precision = model.double_precision
        self._objectives = objectives
        self._parameters = parameters

        self._all_buffers, self._residuals_buffer = self._create_buffers()
        self._kernel = self._build_kernel(self._get_kernel_source(), compile_flags)

    def __del__(self):
        for buffer in self._all_buffers:
            buffer.release()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start
        self._kernel.get_objectives(self._cl_run_context.queue, (int(nmr_problems), ), None, *self._all_buffers,
                                    global_offset=(int(range_start),))
        self._enqueue_readout(self._residuals_buffer, self._objectives, range_start, range_end)

    def _create_buffers(self):
        objectives_buffer = cl.Buffer(self._cl_run_context.context,
                                  cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                  hostbuf=self._objectives)

        all_buffers = [cl.Buffer(self._cl_run_context.context,
                                 cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                                 hostbuf=self._parameters),
                       objectives_buffer]

        for data in self._model.get_data():
            all_buffers.append(cl.Buffer(self._cl_run_context.context,
                                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data))

        return all_buffers, objectives_buffer

    def _get_kernel_source(self):
        nmr_inst_per_problem = self._model.get_nmr_inst_per_problem()
        nmr_params = self._parameters.shape[1]

        kernel_param_names = ['global mot_float_type* params', 'global mot_float_type* objectives']
        kernel_param_names.extend(self._model.get_kernel_param_names(self._cl_environment.device))

        kernel_source = '''
            #define NMR_INST_PER_PROBLEM ''' + str(nmr_inst_per_problem) + '''
        '''

        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += self._model.get_kernel_data_struct(self._cl_environment.device)
        kernel_source += self._model.get_objective_per_observation_function('getObjectiveInstanceValue')
        kernel_source += '''
            __kernel void get_objectives(
                ''' + ",\n".join(kernel_param_names) + '''
                ){
                    ulong gid = get_global_id(0);
                    ''' + self._model.get_kernel_data_struct_initialization(self._cl_environment.device, 'data') + '''

                    mot_float_type x[''' + str(nmr_params) + '''];
                    for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                    }

                    global mot_float_type* result = objectives + gid * NMR_INST_PER_PROBLEM;
                    for(uint i = 0; i < NMR_INST_PER_PROBLEM; i++){
                        result[i] = getObjectiveInstanceValue((void*)&data, x, i);
                    }
            }
        '''
        return kernel_source
