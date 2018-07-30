import logging
import numpy as np

from mot.cl_function import SimpleCLFunction
from mot.cl_runtime_info import CLRuntimeInfo
from mot.kernel_data import Array, Zeros
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


class AbstractOptimizer(object):

    def __init__(self, patience=1, optimizer_settings=None, cl_runtime_info=None, **kwargs):
        """Create a new optimizer that will minimize the given model using the given environments.

        If the environment is None, a suitable default environment is created.

        Args:
            patience (int): The patience is used in the calculation of how many iterations to iterate the optimizer.
                The exact usage of this value of this parameter may change per optimizer.
            optimizer_options (dict): extra options one can set for the optimization routine. These are routine
                dependent.
        """
        self._cl_runtime_info = cl_runtime_info or CLRuntimeInfo()

        self._optimizer_settings = optimizer_settings or {}
        if not isinstance(self._optimizer_settings, dict):
            raise ValueError('The given optimizer settings is not a dictionary.')

        self.patience = patience or 1

        if 'patience' in self._optimizer_settings:
            self.patience = self._optimizer_settings['patience']
        self._optimizer_settings['patience'] = self.patience

    def set_cl_runtime_info(self, cl_runtime_info):
        """Update the CL runtime information.

        Args:
            cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the new runtime information
        """
        self._cl_runtime_info = cl_runtime_info

    @property
    def optimizer_settings(self):
        """Get the optimization flags and settings set to the optimizer.

        This ordinarily should contain all the extra flags and options set to the optimization routine. Even those
        set additionally in the constructor.

        Returns:
            dict: the optimization options (settings)
        """
        return self._optimizer_settings

    def minimize(self, model, starting_positions):
        """Minimize the given model using the given environments.

        Args:
            model (mot.model_interfaces.OptimizeModelInterface): The model to minimize
            starting_positions (ndarray): The starting starting_positions for every problem in the model, it should be
                an matrix of shape (d, p) with d problems and p parameter.

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

    def __str__(self):
        return np.array_str(self.get_optimization_result())


class SimpleOptimizationResult(OptimizationResults):

    def __init__(self, optimization_results, return_codes):
        """Simple optimization results container which computes some values only when requested.

        Args:
            model (mot.model_interfaces.OptimizeModelInterface): the model we used to get these results
            optimization_results (ndarray): a (d, p) matrix with for every d problems the estimated value for
                every parameter p
            return_codes (ndarray): the return codes as a (d,) vector for every d problems
        """
        self._optimization_results = optimization_results
        self._return_codes = return_codes

    def get_optimization_result(self):
        return self._optimization_results

    def get_return_codes(self):
        return self._return_codes


class AbstractParallelOptimizer(AbstractOptimizer):

    def __init__(self, **kwargs):
        """
        Args:
            optimizer_settings (dict): extra options one can set for the optimization routine. These are routine
                dependent.
        """
        super(AbstractParallelOptimizer, self).__init__(**kwargs)
        self._logger = logging.getLogger(__name__)

    def minimize(self, model, starting_positions):
        if len(starting_positions.shape) < 2:
            starting_positions = starting_positions[..., None]

        nmr_problems = starting_positions.shape[0]
        nmr_parameters = starting_positions.shape[1]

        self._logger.info('Entered optimization routine.')
        self._logger.info('Using MOT version {}'.format(__version__))
        self._logger.info('We will use a {} precision float type for the calculations.'.format(
            'double' if self._cl_runtime_info.double_precision else 'single'))
        for env in self._cl_runtime_info.get_cl_environments():
            self._logger.info('Using device \'{}\'.'.format(str(env)))
        self._logger.info('Using compile flags: {}'.format(self._cl_runtime_info.get_compile_flags()))
        self._logger.info('We will use the optimizer {} '
                          'with optimizer settings {}'.format(self.__class__.__name__,
                                                              self._optimizer_settings))

        self._logger.info('Starting optimization preliminaries')

        if len(starting_positions.shape) < 2:
            starting_positions = starting_positions[..., None]

        all_kernel_data = dict(model.get_kernel_data())
        all_kernel_data.update({
            '_parameters': Array(starting_positions, ctype='mot_float_type', is_writable=True, is_readable=True),
            '_return_codes': Zeros((nmr_problems,), ctype='char', is_readable=False, is_writable=True)
        })
        all_kernel_data.update(self._get_optimizer_kernel_data(model, nmr_parameters, nmr_problems))

        self._logger.info('Finished optimization preliminaries')
        self._logger.info('Starting optimization')

        optimizer_func = self._get_optimizer_function(model, nmr_parameters)
        optimizer_func.evaluate({'data': all_kernel_data}, nmr_instances=nmr_problems,
                                use_local_reduction=False, cl_runtime_info=self._cl_runtime_info)

        self._logger.info('Finished optimization')
        return SimpleOptimizationResult(all_kernel_data['_parameters'].get_data(),
                                        all_kernel_data['_return_codes'].get_data())

    def _get_optimizer_kernel_data(self, model, nmr_params, nmr_problems):
        """Get the kernel data specific to the optimization routines.

        Args:
            model (mot.model_interfaces.OptimizeModelInterface): the model we are optimizing
            nmr_params (int): the number of parameters in the model
            nmr_problems (int): the number of problem instances

        Returns:
            dict: kernel input data elements
        """
        return {}

    def _get_optimizer_function(self, model, nmr_params):
        """Get the optimization kernel function.

        By default this returns a full kernel source using information from _get_optimizer_cl_code()
        and _get_optimizer_call_name().

        Args:
            model (mot.model_interfaces.OptimizeModelInterface): the model we are optimizing
            nmr_params (int): the number of parameters in the model

        Returns:
            mot.cl_function.CLFunction: the optimization function to apply
        """
        cl_extra = ''
        cl_extra += self._get_optimizer_cl_code(model, nmr_params)

        return SimpleCLFunction.from_string('''
            void compute(mot_data_struct* data){
                mot_float_type x[''' + str(nmr_params) + '''];
                for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                    x[i] = data->_parameters[i];
                }
                
                *data->_return_codes = (char) ''' + self._get_optimizer_call_name() + '(' + \
                         ', '.join(self._get_optimizer_call_args()) + ''');
                
                for(uint i = 0; i < ''' + str(nmr_params) + '''; i++){
                    data->_parameters[i] = x[i];
                }   
            }
        ''', cl_extra=cl_extra)

    def _get_optimizer_call_args(self):
        """Get the optimizer calling arguments.

        This is useful if a subclass extended the buffers with an additional buffer.

        Returns:
            list of str: the list of optimizer call arguments
        """
        return ['x', '(void*)data']

    def _get_optimizer_cl_code(self, model, nmr_params):
        """Get the optimization CL code that is called during optimization for each problem.

        This is normally called by the default implementation of _get_ll_calculating_kernel().

        By default this creates a CL function named 'evaluation' that can be called by the optimization routine.

        Args:
            model (mot.model_interfaces.OptimizeModelInterface): the model we are optimizing
            nmr_params (int): the number of parameters in the model

        Returns:
            str: The kernel source for the optimization routine.
        """
        kernel_source = self._get_evaluate_function(model, nmr_params)
        kernel_source += self._get_optimization_function(model, nmr_params)
        return kernel_source

    def _get_evaluate_function(self, model, nmr_params):
        """Get the CL code for the evaluation function. This is called from _get_optimizer_cl_code.

        Implementing optimizers can change this if desired.

        Args:
            model (mot.model_interfaces.OptimizeModelInterface): the model we are optimizing
            nmr_params (int): the number of parameters in the model

        Returns:
            str: the evaluation function.
        """
        objective_function = model.get_objective_function()

        kernel_source = ''
        kernel_source += objective_function.get_cl_code()

        kernel_source += '''
            double evaluate(mot_float_type* x, void* data_void){
                mot_data_struct* data = (mot_data_struct*)data_void;
                return ''' + objective_function.get_cl_function_name() + '''(data, x, 0, 0, 0);
            }
        '''
        return kernel_source

    def _get_optimization_function(self, model, nmr_params):
        """Return the optimization function as a CL string for the implementing optimizer.

        This is a convenience function to avoid boilerplate in implementing _get_optimizer_cl_code().

        Args:
            model (mot.model_interfaces.OptimizeModelInterface): the model we are optimizing
            nmr_params (int): the number of parameters in the model

        Returns:
            str: The optimization routine function
        """
        raise NotImplementedError()

    def _get_optimizer_call_name(self):
        """Get the call name of the optimization routine.

        This name is the name of the function called by the kernel to optimize a single problem.

        Returns:
            str: The function name of the optimization function.
        """
        raise NotImplementedError()
