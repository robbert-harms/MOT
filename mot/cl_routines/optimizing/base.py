import logging
import numpy as np
import pyopencl as cl
from mot.cl_routines.mapping.error_measures import ErrorMeasures
from mot.cl_routines.mapping.residual_calculator import ResidualCalculator
from ...utils import results_to_dict, get_float_type_def
from ...cl_routines.base import CLRoutine
from ...load_balance_strategies import Worker
from ...cl_routines.mapping.codec_runner import CodecRunner
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

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=True, patience=1,
                 optimizer_settings=None, **kwargs):
        """Create a new optimizer that will minimize the given model with the given codec using the given environments.

        If the codec is None it is not used, if the environment is None, a suitable default environment should be
        created.

        Args:
            cl_environments (list of CLEnvironment): a list with the cl environments to use
            load_balancer (LoadBalanceStrategy): the load balance strategy to use
            use_param_codec (boolean): if this minimization should use the parameter codecs (param transformations)
            patience (int): The patience is used in the calculation of how many iterations to iterate the optimizer.
                The exact usage of this value of this parameter may change per optimizer.
            optimizer_options (dict): extra options one can set for the optimization routine. These are routine
                dependent.
        """
        self._optimizer_settings = optimizer_settings or {}
        if not isinstance(self._optimizer_settings, dict):
            raise ValueError('The given optimizer settings is not a dictionary.')

        self._use_param_codec = use_param_codec
        self.patience = patience or 1

        if 'patience' in self._optimizer_settings:
            self.patience = self._optimizer_settings['patience']
        self._optimizer_settings['patience'] = self.patience

        super(AbstractOptimizer, self).__init__(cl_environments, load_balancer, **kwargs)

    @property
    def use_param_codec(self):
        """Check if we will use the codec during optimization

        Returns:
            boolean: True if we will use the codec, false otherwise.
        """
        return self._use_param_codec

    @use_param_codec.setter
    def use_param_codec(self, use_param_codec):
        """Set if we will use the codec during optimization

        Args:
            use_param_codec (boolean): Set the value of use_param_codec.
        """
        self._use_param_codec = use_param_codec

    @property
    def optimizer_settings(self):
        """Get the optimization flags and settings set to the optimizer.

        This ordinarily should contain all the extra flags and options set to the optimization routine. Even those
        set additionally in the constructor.

        Returns:
            dict: the optimization options (settings)
        """
        return self._optimizer_settings

    def minimize(self, model, init_params=None, full_output=False):
        """Minimize the given model with the given codec using the given environments.

        Args:
            model (AbstractModel): The model to minimize, instance of AbstractModel
            init_params (dict): A dictionary containing the results of a previous run, provides the starting point
            full_output (boolean): If set to true, the output is a list with the results and a dictionary
                with other outputs. If false, only return the results per problem instance.

        Returns:
            Either only the results per problem, or a list: (results, {}), depending on if full_output is set.
            Both the results dictionary as well as the extra output  dictionary should return results per
            problem instance.
        """


class AbstractParallelOptimizer(AbstractOptimizer):

    def __init__(self, **kwargs):
        """
        Args:
            optimizer_settings (dict): extra options one can set for the optimization routine. These are routine
                dependent.
        """
        super(AbstractParallelOptimizer, self).__init__(**kwargs)
        self._logger = logging.getLogger(__name__)

    def minimize(self, model, init_params=None, full_output=False):
        self._logger.info('Entered optimization routine.')
        self._logger.info('Using MOT version {}'.format(__version__))
        self._logger.info('We will use a {} precision float type for the calculations.'.format(
            'double' if model.double_precision else 'single'))
        for env in self.load_balancer.get_used_cl_environments(self.cl_environments):
            self._logger.info('Using device \'{}\'.'.format(str(env)))
        self._logger.info('Using compile flags: {}'.format(self.get_compile_flags_list(model.double_precision)))
        self._logger.info('The parameters we will optimize are: {0}'.format(model.get_optimized_param_names()))
        self._logger.info('We will use the optimizer {} '
                          'with optimizer settings {}'.format(self.__class__.__name__,
                                                              self._optimizer_settings))

        self._logger.info('Starting optimization preliminaries')

        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        starting_points = np.require(model.get_initial_parameters(init_params), np_dtype,
                                     requirements=['C', 'A', 'O', 'W'])
        nmr_params = starting_points.shape[1]

        return_codes = np.zeros((starting_points.shape[0],), dtype=np.int8, order='C')

        space_transformer = CodecRunner(cl_environments=self.cl_environments, load_balancer=self.load_balancer)
        if self.use_param_codec:
            starting_points = space_transformer.encode(model, starting_points)
            starting_points = np.require(starting_points, np_dtype, requirements=['C', 'A', 'O', 'W'])

        self._logger.info('Finished optimization preliminaries')
        self._logger.info('Starting optimization')

        workers = self._create_workers(self._get_worker_generator(self, model, starting_points,
                                                                  nmr_params, return_codes,
                                                                  self._optimizer_settings))
        self.load_balancer.process(workers, model.get_nmr_problems())
        del workers

        self._logger.info('Finished optimization')

        self._logger.info('Calling finalize optimization results in the model')
        results = model.finalize_optimization_results(results_to_dict(starting_points,
                                                                      model.get_optimized_param_names()))

        self._logger.info('Optimization finished.')

        if full_output:
            extra_output = {'ReturnCodes': return_codes}
            self._logger.info('Calculating errors measures')
            errors = ResidualCalculator(cl_environments=self.cl_environments, load_balancer=self.load_balancer).\
                calculate(model, results)
            extra_output.update(ErrorMeasures(self.cl_environments, self.load_balancer,
                                              model.double_precision).calculate(errors))
            self._logger.info('Done calculating errors measures')

            return results, extra_output
        return results

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
        self._double_precision = model.double_precision
        self._nmr_params = nmr_params

        self._return_codes = return_codes

        self._use_param_codec = self._parent_optimizer.use_param_codec

        self._starting_points = starting_points
        self._all_buffers, self._params_buffer, self._return_code_buffer = self._create_buffers()
        self._kernel = self._build_kernel(self._parent_optimizer.get_compile_flags_list())

    def __del__(self):
        for buffer in self._all_buffers:
            buffer.release()

    def calculate(self, range_start, range_end):
        nmr_problems = range_end - range_start

        kernel_event = self._kernel.minimize(self._cl_run_context.queue, (nmr_problems, ), None, *self._all_buffers,
                                             global_offset=(range_start,))
        return [
            self._enqueue_readout(self._params_buffer, self._starting_points, range_start, range_end, [kernel_event]),
            self._enqueue_readout(self._return_code_buffer, self._return_codes, range_start, range_end, [kernel_event])
        ]

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

        for data in self._model.get_data():
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
        kernel_source += str(self._model.get_kernel_data_struct(self._cl_environment.device))

        if self._use_param_codec:
            kernel_source += self._model.get_parameter_decode_function('decodeParameters') + "\n"

        kernel_source += self._get_optimizer_cl_code()
        kernel_source += '''
            __kernel void minimize(
                ''' + ",\n".join(self._get_kernel_param_names()) + '''
                ){
                    int gid = get_global_id(0);
        '''

        kernel_source += '''
                    mot_float_type x[''' + str(nmr_params) + '''];
                    for(int i = 0; i < ''' + str(nmr_params) + '''; i++){
                        x[i] = params[gid * ''' + str(nmr_params) + ''' + i];
                    }

                    ''' + self._model.get_kernel_data_struct_initialization(self._cl_environment.device, 'data') + '''
                    return_codes[gid] = (char) ''' + self._get_optimizer_call_name() + '(' + \
                         ', '.join(self._get_optimizer_call_args()) + ''');

                    ''' + ('decodeParameters((void*)&data, x);' if self._use_param_codec else '') + '''

                    for(int i = 0; i < ''' + str(nmr_params) + '''; i++){
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
        kernel_param_names.extend(self._model.get_kernel_param_names(self._cl_environment.device))

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

        This is normally called by the default implementation of _get_kernel_source().

        By default this creates a CL function named 'evaluation' that can be called by the optimization routine.
        This default function takes into account the use of the parameter codec and calls the objective function
        of the model to actually evaluate the model.

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
        kernel_source = ''
        kernel_source += self._model.get_objective_function('calculateObjective')
        if self._use_param_codec:
            kernel_source += '''
                double evaluate(mot_float_type* x, const void* data){
                    mot_float_type x_model[''' + str(self._nmr_params) + '''];
                    for(int i = 0; i < ''' + str(self._nmr_params) + '''; i++){
                        x_model[i] = x[i];
                    }
                    decodeParameters(data, x_model);
                    return calculateObjective(data, x_model);
                }
            '''
        else:
            kernel_source += '''
                double evaluate(mot_float_type* x, const void* data){
                    return calculateObjective(data, x);
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
