from pppe.cl_routines.optimizing.levenberg_marquardt import LevenbergMarquardt
from pppe.cl_routines.filters.gaussian import GaussianFilter
from pppe.cl_routines.filters.mean import MeanFilter
from pppe.cl_routines.filters.median import MedianFilter
from pppe.cl_routines.optimizing.gridsearch import GridSearch
from pppe.cl_routines.optimizing.meta_optimizer import MetaOptimizer
from pppe.cl_routines.optimizing.nmsimplex import NMSimplex
from pppe.cl_routines.optimizing.powell import Powell
from pppe.cl_routines.optimizing.serial_optimizers import SerialBasinHopping, SerialBFGS, SerialLM, SerialNMSimplex, \
    SerialPowell
from pppe.load_balance_strategies import EvenDistribution, RuntimeLoadBalancing, PreferGPU, PreferCPU, \
    PreferSpecificEnvironment

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
    optimizers = [LevenbergMarquardt, GridSearch, Powell, NMSimplex, MetaOptimizer, SerialBasinHopping,
                  SerialBFGS, SerialLM, SerialNMSimplex, SerialPowell]
    return _get_item(name, optimizers, 'optimizers')


def get_smoother_by_name(name):
    """ Get the class by the given name.

    This does not instantiate the class, only returns a reference to it.

    Args:
        name: the name of the smoothing routine we want to return

    Returns:
        class: the class of the smoothing routine requested
    """
    smoothers = [GaussianFilter, MeanFilter, MedianFilter]
    return _get_item(name, smoothers, 'smoothers')


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
        if item.get_pretty_name() == name:
            return item
    raise ValueError('The item with the name {0} could not be found in the {1} factory.'.format(name, factory_type))