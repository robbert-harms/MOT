from mot.cl_routines.optimizing.base import AbstractOptimizer

__author__ = 'Robbert Harms'
__date__ = "2016-11-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MultiStepOptimizer(AbstractOptimizer):

    def __init__(self, optimizers, **kwargs):
        """A meta optimization routine that runs multiple optimizers consecutively.

        This meta optimization routine uses uses the result of each optimization routine as starting point for the
        next optimization routine.

        Args:
            optimizer (list of AbstractOptimizer): the optimization routines to run one after another.
        """
        super(MultiStepOptimizer, self).__init__(**kwargs)
        self.optimizers = optimizers

    def minimize(self, model, init_params=None):
        results = None
        for index, optimizer in enumerate(self.optimizers):
            results = optimizer.minimize(model, init_params=init_params)
            init_params = results.get_optimization_result()
        return results
