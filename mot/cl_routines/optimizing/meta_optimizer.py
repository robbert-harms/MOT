import logging

from mot.cl_routines.mapping.calculate_model_estimates import CalculateModelEstimates
from ...cl_routines.optimizing.base import AbstractOptimizer
from ...cl_routines.mapping.error_measures import ErrorMeasures
from ...cl_routines.mapping.residual_calculator import ResidualCalculator
from ...cl_routines.optimizing.nmsimplex import NMSimplex

__author__ = 'Robbert Harms'
__date__ = "2014-06-19"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MetaOptimizer(AbstractOptimizer):

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=True, patience=None, **kwargs):
        """Adds additional functionality to optimization.

        This also calculates error maps for the final fitted model parameters.

        Args:
            cl_environments (list of CLEnvironment): a list with the cl environments to use
            load_balancer (LoadBalancer): the load balance strategy to use
            use_param_codec (boolean): if this minimization should use the parameter codecs (param transformations)
            patience (int): The patience is used in the calculation of how many iterations to iterate the optimizer.
                The exact semantical value of this parameter may change per optimizer.

        Attributes:
            extra_optim_runs (boolean, default 1): The amount of extra optimization runs with a perturbation step.
            extra_optim_runs_optimizers (list, default None): A list of optimizers with one optimizer for every extra
                optimization run. If the length of this list is smaller than the number of runs, the last optimizer is
                used for all remaining runs.
            optimizer (Optimizer, default NMSimplex): The default optimization routine
            add_model_estimates (boolean): if true we add the model estimates to the dictionary returned when
                full_output is set to True
        """
        super(MetaOptimizer, self).__init__(cl_environments, load_balancer, use_param_codec, **kwargs)
        self.enable_sampling = False

        self.add_model_estimates = False
        self.extra_optim_runs = 0
        self.extra_optim_runs_optimizers = []

        self.optimizer = NMSimplex(self.cl_environments, self.load_balancer, use_param_codec=self.use_param_codec,
                                   patience=patience)

        self._propagate_property('cl_environments', cl_environments)
        self._propagate_property('load_balancer', load_balancer)

        self._logger = logging.getLogger(__name__)

    def minimize(self, model, init_params=None, full_output=False):
        results = init_params

        results, extra_maps = self.optimizer.minimize(model, init_params=results, full_output=True)

        if self.extra_optim_runs:
            for i in range(self.extra_optim_runs):
                optimizer = self.optimizer

                if self.extra_optim_runs_optimizers and i < len(self.extra_optim_runs_optimizers):
                    optimizer = self.extra_optim_runs_optimizers[i]

                perturbed_params = model.perturbate(results)
                results, extra_maps = optimizer.minimize(model, init_params=perturbed_params, full_output=True)

        if full_output:
            extra_output = {}

            model_estimates = None
            if self.add_model_estimates:
                self._logger.info('Calculating model estimates')
                model_estimates = CalculateModelEstimates(self.cl_environments,
                                                          self.load_balancer).calculate(model, results)
                extra_output.update({'SignalEstimates': model_estimates})
                self._logger.info('Done calculating model estimates')

            self._logger.info('Calculating errors measures')
            errors = ResidualCalculator(cl_environments=self.cl_environments, load_balancer=self.load_balancer).\
                calculate(model, results, model_estimates=model_estimates)

            extra_output.update(ErrorMeasures(self.cl_environments, self.load_balancer,
                                              model.double_precision).calculate(errors))
            self._logger.info('Done calculating errors measures')

            extra_output.update(extra_maps)
            return results, extra_output

        return results

    @property
    def cl_environments(self):
        return self._cl_environments

    @property
    def load_balancer(self):
        return self._load_balancer

    @cl_environments.setter
    def cl_environments(self, cl_environments):
        self._propagate_property('cl_environments', cl_environments)
        self._cl_environments = cl_environments

    @load_balancer.setter
    def load_balancer(self, load_balancer):
        self._propagate_property('load_balancer', load_balancer)
        self._load_balancer = load_balancer

    def _propagate_property(self, name, value):
        self.optimizer.__setattr__(name, value)

        if self.extra_optim_runs_optimizers:
            for optim in self.extra_optim_runs_optimizers:
                optim.__setattr__(name, value)
