from mot.cl_routines.sampling.metropolis_hastings import MetropolisHastings
from .cl_routines.optimizing.simulated_annealing import SimulatedAnnealing
from .cl_routines.optimizing.levenberg_marquardt import LevenbergMarquardt
from .cl_routines.filters.gaussian import GaussianFilter
from .cl_routines.filters.mean import MeanFilter
from .cl_routines.filters.median import MedianFilter
from .cl_routines.optimizing.nmsimplex import NMSimplex
from .cl_routines.optimizing.powell import Powell
from .load_balance_strategies import EvenDistribution, RuntimeLoadBalancing, PreferGPU, PreferCPU, \
    PreferSpecificEnvironment
from mot.cl_routines.optimizing.praxis import PrAxis

__author__ = 'Robbert Harms'
__date__ = "2015-07-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_optimizer_by_name(name):
    """ Get the class by the given name.

    This does not instantiate the class, only returns a reference to it.

    Args:
        name: the name of the optimizer we want to return

    Returns:
        class: the class of the optimizer requested
    """
    optimizers = [LevenbergMarquardt, Powell, NMSimplex, SimulatedAnnealing, PrAxis]
    return _get_item(name, optimizers, 'optimizers')


def get_sampler_by_name(name):
    """ Get the class by the given name.

    This does not instantiate the class, only returns a reference to it.

    Args:
        name: the name of the optimizer we want to return

    Returns:
        class: the class of the sampler requested
    """
    samplers = [MetropolisHastings]
    return _get_item(name, samplers, 'samplers')


def get_filter_by_name(name):
    """ Get the class by the given name.

    This does not instantiate the class, only returns a reference to it.

    Args:
        name: the name of the filter routine we want to return

    Returns:
        class: the class of the filter routine requested
    """
    filters = [GaussianFilter, MeanFilter, MedianFilter]
    return _get_item(name, filters, 'smoothers')


def get_load_balance_strategy_by_name(name):
    """ Get the class by the given name.

    This does not instantiate the class, only returns a reference to it.

    Args:
        name: the name of the load balance strategy we want to return

    Returns:
        class: the class of the load balance strategy requested
    """
    lb = [EvenDistribution, RuntimeLoadBalancing, PreferGPU, PreferCPU, PreferSpecificEnvironment]
    return _get_item(name, lb, 'load balancers')


def _get_item(name, item_list, factory_type):
    for item in item_list:
        if item.__name__ == name:
            return item
    raise ValueError('The item with the name {0} could not be found in the {1} factory.'.format(name, factory_type))
