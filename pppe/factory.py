from pppe.cl_routines.smoothing.gaussian import GaussianSmoother
from pppe.cl_routines.smoothing.mean import MeanSmoother
from pppe.cl_routines.smoothing.median import MedianSmoother
from pppe.cl_routines.optimizing.gridsearch import GridSearch
from pppe.cl_routines.optimizing.meta_optimizer import MetaOptimizer
from pppe.cl_routines.optimizing.nmsimplex import NMSimplex
from pppe.cl_routines.optimizing.powell import Powell
from pppe.cl_routines.optimizing.serial_optimizers import SerialBasinHopping, SerialBFGS, SerialLM, SerialNMSimplex, \
    SerialPowell

__author__ = 'Robbert Harms'
__date__ = "2015-07-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def get_optimizer_by_name(name):
    """ Get the class of the optimizer by the given name.

    This does not instantiate the class, only returns a reference to it.

    Args:
        name: the name of the optimizer we want to return
    """
    if name == 'GridSearch':
        return GridSearch
    elif name == 'Powell':
        return Powell
    elif name == 'NMSimplex':
        return NMSimplex
    elif name == 'MetaOptimizer':
        return MetaOptimizer
    elif name == 'SerialBasinHopping':
        return SerialBasinHopping
    elif name == 'SerialBFGS':
        return SerialBFGS
    elif name == 'SerialLM':
        return SerialLM
    elif name == 'SerialNMSimplex':
        return SerialNMSimplex
    elif name == 'SerialPowell':
        return SerialPowell


def get_smoother_by_name(name):
    """ Get the class of the optimizer by the given name.

    This does not instantiate the class, only returns a reference to it.

    Args:
        name: the name of the optimizer we want to return
    """
    if name == 'Gaussian':
        return GaussianSmoother
    elif name == 'Mean':
        return MeanSmoother
    elif name == 'Median':
        return MedianSmoother