import logging
import numpy as np
import pyopencl as cl

from mot.cl_routines.mapping.error_measures import ErrorMeasures
from mot.cl_routines.mapping.residual_calculator import ResidualCalculator
from ...utils import get_float_type_def
from ...cl_routines.base import CLRoutine
from ...load_balance_strategies import Worker
from ...__version__ import __version__

__author__ = 'Robbert Harms'
__date__ = "2014-05-18"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


return_code_labels = {
    0: ['default', 'no return code specified'],
    1: ['found zero', 'sum of squares/evaluation below underflow limit'],
    2: ['converged', 'the relative error in the sum of squares/evaluation is at most tol'],
    3: ['converged', 'the relative error of the parameter vector is at most tol'],
    4: ['converged', 'both errors are at most tol'],
    5: ['trapped', 'by degeneracy; increasing epsilon might help'],
    6: ['exhausted', 'number of function calls exceeding preset patience'],
    7: ['failed', 'ftol<tol: cannot reduce sum of squares any further'],
    8: ['failed', 'xtol<tol: cannot improve approximate solution any further'],
    9: ['failed', 'gtol<tol: cannot improve approximate solution any further'],
    10: ['NaN', 'Function value is not-a-number or infinite'],
    11: ['exhausted', 'temperature decreased to 0.0']
}


class AbstractOptimizer(CLRoutine):

    def __init__(self, cl_environments=None, load_balancer=None, patience=1, optimizer_settings=None, **kwargs):
        """Create a new optimizer that will minimize the given model using the given environments.

        If the environment is None, a suitable default environment is created.

        Args:
            cl_environments (list of CLEnvironment): a list with the cl environments to use
            load_balancer (SimpleLoadBalanceStrategy): the load balance strategy to use
            patience (int): The patience is used in the calculation of how many iterations to iterate the optimizer.
                The exact usage of this value of this parameter may change per optimizer.
            optimizer_options (dict): extra options one can set for the optimization routine. These are routine
                dependent.
        """
        self._optimizer_settings = optimizer_settings or {}
        if not isinstance(self._optimizer_settings, dict):
            raise ValueError('The given optimizer settings is not a dictionary.')

        self.patience = patience or 1

        if 'patience' in self._optimizer_settings:
            self.patience = self._optimizer_settings['patience']
        self._optimizer_settings['patience'] = self.patience

        super(AbstractOptimizer, self).__init__(cl_environments, load_balancer, **kwargs)

    @property
    def optimizer_settings(self):
        """Get the optimization flags and settings set to the optimizer.

        This ordinarily should contain all the extra flags and options set to the optimization routine. Even those
        set additionally in the constructor.

        Returns:
            dict: the optimization options (settings)
        """
        return self._optimizer_settings

    def minimize(self, model, init_params=None):
        """Minimize the given model using the given environments.

        Args:
            model (AbstractModel): The model to minimize, instance of AbstractModel
            init_params (ndarray): A starting point for every problem in the model, if not set we take
                the default from the model itself. If given, it should be an matrix of shape (d, p) with d problems and
                p parameter starting points.

        Returns:
            OptimizationResults: the container with the optimization results
        """
        raise NotImplementedError()


class OptimizationResults(object):

    def get_optimization_result(self):
        """Get the optimization result, that is, the matrix with the optimized parameter values.

        Returns:
            ndarray: the optimized parameter maps, an (d, p) array with for d problems a value for every p parameters
        """
        raise NotImplementedError()

    def get_return_codes(self):
        """Get the return codes, that is, the matrix with for every problem the return code of the optimizer

        Returns:
            ndarray: the return codes, an (d,) vector with for d problems a return code
        """
        raise NotImplementedError()

    def get_residuals(self):
        """Get the residuals per problem instance.

        Returns:
            ndarray: (d, r) matrix with for d problems r residuals
        """
        raise NotImplementedError()

    def get_error_measures(self):
        """Get some error measures.

        Returns:
            dict: a dictionary with (d,*) matrices with interesting error measures.
                The first dimension of every return matrix is of length d (for d problems). The other dimensions may
                vary.
        """
        raise NotImplementedError()


class SimpleOptimizationResult(OptimizationResults):

    def __init__(self, model, optimization_results, return_codes):
        """Simple optimization results container which computes some values only when requested.

        Args:
            model (mot.model_interfaces.OptimizeModelInterface): the model we used to get these results
            optimization_results (ndarray): a (d, p) matrix with for every d problems and p parameters the estimated
                value
            return_codes (ndarray): the return codes as a (d,) vector for every d problems
        """
        self._model = model
        self._optimization_results = optimization_results
        self._return_codes = return_codes
        self._error_measures = None

    def get_optimization_result(self):
        return self._optimization_results

    def get_return_codes(self):
        return self._return_codes

    def get_residuals(self):
        return np.nan_to_num(ResidualCalculator().calculate(self._model, self._optimization_results))

    def get_error_measures(self):
        if self._error_measures is None:
            self._error_measures = ErrorMeasures(
                double_precision=self._model.double_precision).calculate(self.get_residuals())
        self._error_measures = {k: np.nan_to_num(v) for k, v in self._error_measures.items()}
        return self._error_measures


class AbstractParallelOptimizer(AbstractOptimizer):

    def __init__(self, **kwargs):
        """
        Args:
            optimizer_settings (dict): extra options one can set for the optimization routine. These are routine
                dependent.
        """
        super(AbstractParallelOptimizer, self).__init__(**kwargs)
        self._logger = logging.getLogger(__name__)

    def minimize(self, model, init_params=None):
        self._logger.info('Entered optimization routine.')
        self._logger.info('Using MOT version {}'.format(__version__))
        self._logger.info('We will use a {} precision float type for the calculations.'.format(
            'double' if model.double_precision else 'single'))
        for env in self.load_balancer.get_used_cl_environments(self.cl_environments):
            self._logger.info('Using device \'{}\'.'.format(str(env)))
        self._logger.info('Using compile flags: {}'.format(self.get_compile_flags_list(model.double_precision)))
        self._logger.info('The parameters we will optimize are: {0}'.format(model.get_free_param_names()))
        self._logger.info('We will use the optimizer {} '
                          'with optimizer settings {}'.format(self.__class__.__name__,
                                                              self._optimizer_settings))

        self._logger.info('Starting optimization preliminaries')

        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        if init_params is None:
            init_params = model.get_initial_parameters()

        starting_points = np.require(init_params, np_dtype,
                                     requirements=['C', 'A', 'O', 'W'])

        nmr_params = starting_points.shape[1]

        return_codes = np.zeros((starting_points.shape[0],), dtype=np.int8, order='C')

        self._logger.info('Finished optimization preliminaries')
        self._logger.info('Starting optimization')

        workers = self._create_workers(self._get_worker_generator(self, model, starting_points,
                                                                  nmr_params, return_codes,
                                                                  self._optimizer_settings))
        self.load_balancer.process(workers, model.get_nmr_problems())

        self._logger.info('Finished optimization')

        return SimpleOptimizationResult(model, starting_points, return_codes)

    def _get_worker_generator(self, *args):
        """Generate the worker generator callback for the function _create_workers()

        This is supposed to be overwritten by the implementing optimizer.

        Returns:
            the python callback for generating the worker
        """
        return lambda cl_environment: AbstractParallelOptimizerWorker(cl_environment, *args)


class AbstractParallelOptimizerWorker(Worker):

    def __init__(self, cl_environment, parent_optimizer, model, starting_points,
                 nmr_params, return_codes, optimizer_settings=None):
        super(AbstractParallelOptimizerWorker, self).__init__(cl_environment)

        self._optimizer_settings = optimizer_settings

        self._parent_optimizer = parent_optimizer

        self._model = model
        self._data_info = self._model.get_kernel_data_info()
        self._double_precision = model.double_precision
        self._nmr_params = nmr_params

        self._return_codes = return_codes

        self._starting_points = starting_points
        self._all_buffers, self._params_buffer, self._return_code_buffer = self._create_buffers()
        self._kernel = self._build_kernel(self._get_kernel_source(), self._parent_optimizer.get_compile_flags_list())

    def __del__(self):
        for buffer in self._all_buffers:
            buffer.release()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        self._kernel.minimize(self._cl_run_context.queue, (nmr_problems, ), None, *self._all_buffers,
                              global_offset=(range_start,))
        self._enqueue_readout(self._params_buffer, self._starting_points, range_start, range_end)
        self._enqueue_readout(self._return_code_buffer, self._return_codes, range_start, range_end)

    def _create_buffers(self):
        all_buffers = []

        parameters_buffer = cl.Buffer(self._cl_run_context.context,
                                      cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                                      hostbuf=self._starting_points)
        all_buffers.append(parameters_buffer)

        return_code_buffer = cl.Buffer(self._cl_run_context.context,
                                       cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                                       hostbuf=self._return_codes)
        all_buffers.append(return_code_buffer)

        for data in self._data_info.get_data():
            all_buffers.append(cl.Buffer(self._cl_run_context.context,
                                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data))

        return all_buffers, parameters_buffer, return_code_buffer

    def _get_kernel_source(self):
        """Generate the kernel source for this optimization routine.

        By default this returns a full kernel source using information from _get_optimizer_cl_code()
        and _get_optimizer_call_name().

        One could overwrite this function to completely generate the kernel source, but most likely
        you would want to implement _get_optimizer_cl_code() and _get_optimizer_call_name().

        Returns:
            str: The kernel source for this optimization routine.
        """
        nmr_params = self._nmr_params

        kernel_source = ''
        kernel_source += get_float_type_def(self._double_precision)
        kernel_source += str(self._data_info.get_kernel_data_struct())

        kernel_source += self._get_optimizer_cl_code()
        kernel_source += '''
            __kernel void minimize(
                ''' + ",\n".join(self._get_kernel_param_names()) + '''
                ){
                    ulong gid = get_global_id(0);
        '''

        kernel_source += '''
                    mot_float_type x[''' + str(nmr_params) + '''];
                    for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                    }

                    ''' + self._data_info.get_kernel_data_struct_initialization('data') + '''
                    return_codes[gid] = (char) ''' + self._get_optimizer_call_name() + '(' + \
                         ', '.join(self._get_optimizer_call_args()) + ''');

                    for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                        params[gid * ''' + str(nmr_params) + ''' + i] = x[i];
                    }
                }
        '''
        return kernel_source

    def _get_kernel_param_names(self):
        """Get the list of kernel parameter names.

        This is useful if a subclass extended the buffers with an additional buffer.

        Returns:
            list of str: the list of kernel parameter names
        """
        kernel_param_names = ['global mot_float_type* params',
                              'global char* return_codes']
        kernel_param_names.extend(self._data_info.get_kernel_parameters())

        return kernel_param_names

    def _get_optimizer_call_args(self):
        """Get the optimizer calling arguments.

        This is useful if a subclass extended the buffers with an additional buffer.

        Returns:
            list of str: the list of optimizer call arguments
        """
        call_args = ['x', '(const void*) &data']
        return call_args

    def _get_optimizer_cl_code(self):
        """Get the optimization CL code that is called during optimization for each voxel.

        This is normally called by the default implementation of _get_ll_calculating_kernel().

        By default this creates a CL function named 'evaluation' that can be called by the optimization routine.

        Returns:
            str: The kernel source for the optimization routine.
        """
        kernel_source = self._get_evaluate_function()
        kernel_source += self._get_optimization_function()
        return kernel_source

    def _get_evaluate_function(self):
        """Get the CL code for the evaluation function. This is called from _get_optimizer_cl_code.

        Implementing optimizers can change this if desired.

        Returns:
            str: the evaluation function.
        """
        objective_function = self._model.get_objective_per_observation_function()
        param_modifier = self._model.get_pre_eval_parameter_modifier()

        kernel_source = ''
        kernel_source += objective_function.get_function()
        kernel_source += param_modifier.get_function()
        kernel_source += '''
            double evaluate(mot_float_type* x, const void* data){
                
                mot_float_type x_model[''' + str(self._model.get_nmr_estimable_parameters()) + '''];
                for(uint i = 0; i < ''' + str(self._model.get_nmr_estimable_parameters()) + '''; i++){
                    x_model[i] = x[i];
                }
                ''' + param_modifier.get_name() + '''((void*)&data, x_model);
                
                double sum = 0;
                for(uint i = 0; i < ''' + str(self._model.get_nmr_inst_per_problem()) + '''; i++){
                    sum += pown(''' + objective_function.get_name() + '''(data, x_model, i), 2);
                }
                return sum;
            }
        '''
        return kernel_source

    def _get_optimization_function(self):
        """Return the optimization function as a CL string for the implementing optimizer.

        This is a convenience function to avoid boilerplate in implementing _get_optimizer_cl_code().

        Returns:
            str: The optimization routine function
        """

    def _get_optimizer_call_name(self):
        """Get the call name of the optimization routine.

        This name is the name of the function called by the kernel to optimize a single voxel.

        Returns:
            str: The function name of the optimization function.
        """
        return ''
