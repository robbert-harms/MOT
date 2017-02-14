import pyopencl as cl
import numpy as np

from mot.utils import is_scalar
from ...cl_routines.optimizing.base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GridGenerator(object):

    def get_grid(self, model, grid_size):
        """Get the parameter grid we use to evaluate the model against.

        Args:
            model: the model for which to generate or return a parameter grid
            grid_size: an indication of the desired grid size, the grid may be larger

        Returns:
            ndarray: a grid of parameters with for every parameter (column) an starting point (row). That is,
                every row contains a unique starting point.
        """
        return NotImplementedError


class GivenGrid(GridGenerator):

    def __init__(self, grid):
        """Use the given grid for the computations.

        Args:
            grid (ndarray): the grid we use for the computations. The second dimension should match the number
                of parameters in the model.
        """
        self._grid = grid

    def get_grid(self, model, grid_size):
        return self._grid


class LinearSpacedGrid(GridGenerator):

    def __init__(self, grid_nmr_steps=None):
        """Creates a linear spaced grid for the grid search.

        The steps per dimension can be controlled using the parameter ``grid_nmr_steps``. If this is a scalar
        we use that for all dimensions, if it is a list we use the steps given for each dimension separately.

        Setting a grid step to 1 or lower disables that dimension from the grid search and the average
        initial parameter value is returned instead.

        If the grid number of steps is not given we use the 'grid_size' argument in the method 'get_grid' for
        generating the grid (typically set to the patience of the optimizer). The total grid size in that case is then
        ``grid_size * nmr_parameters``.

        Args:
            grid_nmr_steps (int or list): the number of steps for the generated grid. If set to None we
                use the argument 'grid_size' of the method 'get_grid' for generating the grid.
        """
        self._grid_nmr_steps = grid_nmr_steps

    def get_grid(self, model, grid_size):
        return self._generate_grid(model, grid_size)

    def _generate_grid(self, model, grid_size):
        """Generate a grid for the grid search

        Returns:
            ndarray: a nxm array where n is the number of grid points and m is the number of parameters
        """
        spacings = self._get_grid_spacing(model, grid_size)

        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        lower_bounds = model.get_lower_bounds()
        upper_bounds = model.get_upper_bounds()
        default_params = np.mean(model.get_initial_parameters(), axis=0)

        result = np.zeros((1, model.get_nmr_estimable_parameters()), np_dtype)

        repeat_mult = 1
        for index, spacing in enumerate(spacings):
            if spacing <= 1:
                spacing = 1
                values = default_params[index]
            else:
                values = np.linspace(np.min(lower_bounds[index]), np.max(upper_bounds[index]), spacing)

            result = np.tile(result, (spacing, 1))
            result[:, index] = np.repeat(values, repeat_mult)

            repeat_mult *= spacing

        return result

    def _get_grid_spacing(self, model, grid_size):
        """This returns the grid spacings used to generate a grid automatically.

        Returns:
            list: the grid spacings with one integer for every dimension.
        """
        if self._grid_nmr_steps is None:
            return [np.round(grid_size ** (1.0/model.get_nmr_estimable_parameters()))] \
                        * model.get_nmr_estimable_parameters()
        elif is_scalar(self._grid_nmr_steps):
            return [self._grid_nmr_steps] * model.get_nmr_estimable_parameters()

        if len(self._grid_nmr_steps) != model.get_nmr_estimable_parameters():
            raise ValueError('The length of the given grid ({}) steps does '
                             'not equal the number of parameters ({}).'.format(self._grid_nmr_steps,
                                                                               model.get_nmr_estimable_parameters()))

        return self._grid_nmr_steps


class UniformRandomGrid(GridGenerator):

    def __init__(self):
        """Creates an uniform random grid for the grid search."""

    def get_grid(self, model, grid_size):
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        result = np.zeros((grid_size, model.get_nmr_estimable_parameters()), np_dtype)

        lower_bounds = model.get_lower_bounds()
        upper_bounds = model.get_upper_bounds()

        for param_ind in range(len(lower_bounds)):
            result[:, param_ind] = np.random.uniform(np.min(lower_bounds[param_ind]),
                                                     np.max(upper_bounds[param_ind]), grid_size)

        return result


class GaussianRandomGrid(GridGenerator):

    def __init__(self):
        """Generates a random parameter grid using a normal distribution around the current initial value.

        This uses the mean of the initial value as the mean of the distribution and half the difference
        between the lower and upper bounds as the std.
        """

    def get_grid(self, model, grid_size):
        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        result = np.zeros((grid_size, model.get_nmr_estimable_parameters()), np_dtype)

        lower_bounds = model.get_lower_bounds()
        upper_bounds = model.get_upper_bounds()
        default_params = np.mean(model.get_initial_parameters(), axis=0)

        for param_ind in range(len(lower_bounds)):
            result[:, param_ind] = np.random.normal(
                default_params[param_ind],
                (np.max(upper_bounds[param_ind]) - np.min(lower_bounds[param_ind])) / 2.0,
                grid_size)

        return result


class GridSearch(AbstractParallelOptimizer):

    default_patience = 250

    def __init__(self, grid_generator=None, patience=None, **kwargs):
        """A grid search routine that searches an grid for a good solution.

        This optimization routine evaluates the model at every point in a grid (the same points for every
        problem instance) and retunrs per problem instance the parameters with the lowest function value.

        Please note that the current starting point is always evaluated as well and can be returned as optimum for a
        problem instance.

        For generating the grid we use an instance of an :class:`ParameterGrid`.

        Args:
            grid_generator (:class:`GridGenerator`): the parameter grid to use
        """
        self._grid_generator = grid_generator
        patience = patience or self.default_patience
        super(GridSearch, self).__init__(patience=patience, **kwargs)

    def _get_worker_generator(self, *args):
        model = args[1]
        nmr_params = args[3]

        np_dtype = np.float32
        if model.double_precision:
            np_dtype = np.float64

        grid = np.require(self._grid_generator.get_grid(
            model, self.patience * nmr_params), np_dtype, requirements=['C', 'A', 'O'])

        if grid.shape[1] != model.get_nmr_estimable_parameters():
            raise ValueError('The shape of the generated grid is not compatible with the given model.')

        return lambda cl_environment: GridSearchWorker(cl_environment, *args, grid=grid)


class GridSearchWorker(AbstractParallelOptimizerWorker):

    def __init__(self, *args, **kwargs):
        self._grid = kwargs.pop('grid')
        super(GridSearchWorker, self).__init__(*args, **kwargs)

    def _create_buffers(self):
        buffers = super(GridSearchWorker, self)._create_buffers()

        parameters_buffer = cl.Buffer(self._cl_run_context.context,
                                      cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                                      hostbuf=self._grid)
        buffers[0].append(parameters_buffer)
        return buffers

    def _get_kernel_param_names(self):
        kernel_param_names = super(GridSearchWorker, self)._get_kernel_param_names()
        kernel_param_names.append('global mot_float_type* grid')
        return kernel_param_names

    def _get_optimizer_call_args(self):
        call_args = super(GridSearchWorker, self)._get_optimizer_call_args()
        call_args.append('grid')
        return call_args

    def _get_optimization_function(self):
        nmr_params = self._nmr_params

        kernel_source = ''
        kernel_source += self._model.get_parameter_encode_function('encodeParameters') + "\n"

        kernel_source += '''
            int grid_search(mot_float_type* model_parameters, const void* const data, global mot_float_type* grid){

                int param_ind;

                mot_float_type best_params_so_far[''' + str(nmr_params) + '''];
                for (param_ind = 0; param_ind < ''' + str(nmr_params) + '''; param_ind++){
                    best_params_so_far[param_ind] = model_parameters[param_ind];
                }

                mot_float_type lowest_error = evaluate(model_parameters, data);
                mot_float_type error = 0.0;

                for(int i = 0; i < ''' + str(self._grid.shape[0]) + '''; i++){

                    for (param_ind = 0; param_ind < ''' + str(nmr_params) + '''; param_ind++){
                        model_parameters[param_ind] = grid[i * ''' + str(nmr_params) + ''' + param_ind];
                    }

                    encodeParameters(model_parameters, (void*)&data);
                    error = evaluate(model_parameters, data);

                    if(error < lowest_error){
                        lowest_error = error;

                        for (param_ind = 0; param_ind < ''' + str(nmr_params) + '''; param_ind++){
                            best_params_so_far[param_ind] = model_parameters[param_ind];
                        }
                    }
                }

                for (param_ind = 0; param_ind < ''' + str(nmr_params) + '''; param_ind++){
                    model_parameters[param_ind] = best_params_so_far[param_ind];
                }

                return 0;
            }
        '''
        return kernel_source

    def _get_optimizer_call_name(self):
        return 'grid_search'
