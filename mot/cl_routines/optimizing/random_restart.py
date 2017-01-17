import numpy as np
from mot.cl_routines.optimizing.base import AbstractOptimizer

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
            previous_results (dict): the dictionary with per parameter the results

        Returns:
            dict: per parameter a ndarray with the new starting points per problem instance.
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

        init_dict = {}

        for param_ind, param_name in enumerate(model.get_optimized_param_names()):
            init_dict[param_name] = np.ones(model.get_nmr_problems()) * self.grid[self._iteration_counter, param_ind]

        self._iteration_counter += 1
        return init_dict


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

        param_names = model.get_optimized_param_names()
        lower_bounds = model.get_lower_bounds()
        upper_bounds = model.get_upper_bounds()

        init_dict = {}

        for param_ind, param_name in enumerate(param_names):
            init_dict[param_name] = np.ones(model.get_nmr_problems()) * \
                                    np.random.uniform(np.min(lower_bounds[param_ind]),
                                                      np.max(upper_bounds[param_ind]))

        return init_dict


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

        init_dict = {}

        for param_ind, param_name in enumerate(model.get_optimized_param_names()):
            std = np.std(previous_results[param_name])

            points = np.zeros(model.get_nmr_problems())
            for problem_ind in range(model.get_nmr_problems()):
                points[problem_ind] = np.random.normal(previous_results[param_name][problem_ind], std)

            init_dict[param_name] = points

        return init_dict


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
            starting_point_generator (StartingPointGenerator): the randomizer instance we use to randomize the starting point
        """
        super(RandomRestart, self).__init__(**kwargs)
        self._optimizer = optimizer
        self._starting_point_generator = starting_point_generator

    def minimize(self, model, init_params=None, full_output=False):
        results = self._optimizer.minimize(model, init_params, full_output=True)

        starting_points = self._starting_point_generator.next(model, results[0])
        while starting_points:
            new_results = self._optimizer.minimize(model, starting_points, full_output=True)

            results = self._get_best_results(results, new_results)

            starting_points = self._starting_point_generator.next(model, results[0])

        return results

    def _get_best_results(self, previous_results, new_results):
        result_choice = np.argmin([previous_results[1]['Errors.l2'], new_results[1]['Errors.l2']], axis=0)

        output = []
        for results_ind in range(len(previous_results)):
            output_dict = {}

            for map_name in previous_results[results_ind]:
                output_dict[map_name] = np.array([previous_results[results_ind][map_name],
                          new_results[results_ind][map_name]])[(result_choice, range(result_choice.shape[0]))]

            output.append(output_dict)

        return output
