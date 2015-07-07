from pppe.cl_routines.smoothing.gaussian import GaussianSmoother
from pppe.cl_routines.smoothing.mean import MeanSmoother
from pppe.cl_routines.smoothing.median import MedianSmoother
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
    if name == 'GridSearch':
        return GridSearch
    if name == 'Powell':
        return Powell
    if name == 'NMSimplex':
        return NMSimplex
    if name == 'MetaOptimizer':
        return MetaOptimizer
    if name == 'SerialBasinHopping':
        return SerialBasinHopping
    if name == 'SerialBFGS':
        return SerialBFGS
    if name == 'SerialLM':
        return SerialLM
    if name == 'SerialNMSimplex':
        return SerialNMSimplex
    if name == 'SerialPowell':
        return SerialPowell


def get_smoother_by_name(name):
    """ Get the class by the given name.

    This does not instantiate the class, only returns a reference to it.

    Args:
        name: the name of the smoothing routine we want to return

    Returns:
        class: the class of the smoothing routine requested
    """
    if name == 'Gaussian':
        return GaussianSmoother
    if name == 'Mean':
        return MeanSmoother
    if name == 'Median':
        return MedianSmoother


def get_load_balance_strategy_by_name(name):
    """ Get the class by the given name.

    This does not instantiate the class, only returns a reference to it.

    Args:
        name: the name of the load balance strategy we want to return

    Returns:
        class: the class of the load balance strategy requested
    """
    if name == 'EvenDistribution':
        return EvenDistribution
    if name == 'RuntimeLoadBalancing':
        return RuntimeLoadBalancing
    if name == 'PreferGPU':
        return PreferGPU
    if name == 'PreferCPU':
        return PreferCPU
    if name == 'PreferSpecificEnvironment':
        return PreferSpecificEnvironment
