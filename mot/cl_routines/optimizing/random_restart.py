import numpy as np
from mot.cl_routines.optimizing.base import AbstractOptimizer, SimpleOptimizationResult

__author__ = 'Robbert Harms'
__date__ = "2016-11-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class StartingPointGenerator(object):

    def next(self, model, previous_results):
        """Returns a next starting point given the model and the previous results.

        This method can return None which means that no next starting point is available.

        Args:
            model: the model for which we are generating points
            previous_results (ndarray): the previous results, an (d, p) array with for every d problems and n parameters
                the (semi-)optimum value

        Returns:
            ndarray: array of same type and shape as the input but with the new starting points
        """
        return NotImplementedError


class PointsFromGrid(StartingPointGenerator):

    def __init__(self, grid):
        """Uses the given points as a starting point for all problem instances per run.

        This accepts a grid with per row a starting point that we will use for all the problem instances for
        that optimization run. The number of rows determines the number of iterations.

        Args:
            grid (ndarray): the grid with one starting point per iteration.
        """
        self.grid = grid
        self._iteration_counter = 0

    def next(self, model, previous_results):
        if self._iteration_counter == self.grid.shape[0]:
            return None

        new_points = np.zeros_like(previous_results)

        for param_ind in range(new_points.shape[1]):
            new_points[:, param_ind] = np.ones(model.get_nmr_problems()) * self.grid[self._iteration_counter, param_ind]

        self._iteration_counter += 1
        return new_points


class UsePrevious(StartingPointGenerator):

    def __init__(self, number_of_runs=3):
        """A strategy that uses the previous results without alterations for the next optimization run.

        Args:
            number_of_runs (int): the number of times we optimize using this strategy
        """
        self.number_of_runs = number_of_runs
        self._iteration_counter = 0

    def next(self, model, previous_results):
        if self._iteration_counter == self.number_of_runs:
            return None
        self._iteration_counter += 1
        return previous_results


class RandomStartingPoint(StartingPointGenerator):

    def __init__(self, number_of_runs=3):
        """A strategy that generates uniformly random starting points for every parameter.

        Per run this generates for each parameter a uniformly distributed random number between the lower
        and upper bounds and uses that single random value for all problem instances.

        Hence, this does not generate a unique random point per problem instance, but uses a single random
        point for all problem instances per iteration.

        Args:
            number_of_runs (int): the number of times we optimize using this strategy
        """
        self.number_of_runs = number_of_runs
        self._iteration_counter = 0

    def next(self, model, previous_results):
        if self._iteration_counter == self.number_of_runs:
            return None
        self._iteration_counter += 1

        lower_bounds = model.get_lower_bounds()
        upper_bounds = model.get_upper_bounds()

        new_points = np.zeros_like(previous_results)
        for param_ind in range(new_points.shape[1]):
            new_points[:, param_ind] = np.ones(model.get_nmr_problems()) * \
                                    np.random.uniform(np.min(lower_bounds[param_ind]),
                                                      np.max(upper_bounds[param_ind]))
        return new_points


class GaussianPerturbation(StartingPointGenerator):

    def __init__(self, number_of_runs=3):
        """A strategy that perturbates the previous results using a Normal distribution

        Per run this generates for each parameter and for each problem instance a new starting position using
        the previous parameter as a mean and for standard deviation it uses the standard deviation over the problem
        instances.

        Hence, this generates a unique random point per problem instance, in contrast to some of the other strategies.

        Args:
            number_of_runs (int): the number of times we optimize using this strategy
        """
        self.number_of_runs = number_of_runs
        self._iteration_counter = 0

    def next(self, model, previous_results):
        if self._iteration_counter == self.number_of_runs:
            return None
        self._iteration_counter += 1

        lower_bounds = model.get_lower_bounds()
        upper_bounds = model.get_upper_bounds()

        new_points = np.zeros_like(previous_results)
        for param_ind in range(new_points.shape[1]):
            std = np.std(previous_results[:, param_ind])

            for problem_ind in range(model.get_nmr_problems()):
                new_points[problem_ind, param_ind] = np.clip(
                    np.random.normal(previous_results[problem_ind, param_ind], std),
                    lower_bounds[param_ind], upper_bounds[param_ind])
        return new_points


class RandomRestart(AbstractOptimizer):

    def __init__(self, optimizer, starting_point_generator, **kwargs):
        """A meta optimization routine that allows multiple random restarts.

        This meta optimizer runs the given optimizer multiple times using different starting positions for each run.
        The starting positions are obtained using a starting point generator which returns new starting points
        given the initial starting points.

        Please note that the initial starting point is always optimized first as a baseline reference.

        The returned results contain per problem instance the parameter that resulted in the lowest function value.

        Args:
            optimizer (AbstractOptimizer): the optimization routines to run one after another.
            starting_point_generator (StartingPointGenerator): the randomizer instance we use
                to randomize the starting point
        """
        super(RandomRestart, self).__init__(**kwargs)
        self._optimizer = optimizer
        self._starting_point_generator = starting_point_generator

    def minimize(self, model, init_params=None):
        opt_output = self._optimizer.minimize(model, init_params)
        l2_errors = opt_output.get_error_measures()['Errors.l2']
        results = opt_output.get_optimization_result()
        return_codes = opt_output.get_return_codes()

        starting_points = self._starting_point_generator.next(model, results)
        while starting_points is not None:
            new_opt_output = self._optimizer.minimize(model, starting_points)
            new_results = new_opt_output.get_optimization_result()
            new_l2_errors = new_opt_output.get_error_measures()['Errors.l2']
            return_codes = new_opt_output.get_return_codes()

            results, l2_errors = self._get_best_results(results, new_results, l2_errors, new_l2_errors)

            starting_points = self._starting_point_generator.next(model, results)

        return SimpleOptimizationResult(model, results, return_codes)

    def _get_best_results(self, previous_results, new_results, previous_l2_errors, new_l2_errors):
        result_choice = np.argmin([previous_l2_errors, new_l2_errors], axis=0)

        results = np.zeros_like(previous_results)

        for param_ind in range(previous_results.shape[1]):
            choices = np.array([previous_results[:, param_ind], new_results[:, param_ind]])
            results[:, param_ind] = choices[(result_choice, range(result_choice.shape[0]))]

        resulting_l2_errors = np.array([previous_l2_errors, new_l2_errors])[(result_choice,
                                                                             range(result_choice.shape[0]))]

        return results, resulting_l2_errors
