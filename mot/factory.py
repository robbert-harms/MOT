from mot.sample import AdaptiveMetropolisWithinGibbs
from mot.sample import SingleComponentAdaptiveMetropolis

__author__ = 'Robbert Harms'
__date__ = "2015-07-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


samplers = [AdaptiveMetropolisWithinGibbs, SingleComponentAdaptiveMetropolis]


def get_sampler_by_name(name):
    """ Get the class by the given name.

    This does not instantiate the class, only returns a reference to it.

    Args:
        name: the name of the optimizer we want to return

    Returns:
        class: the class of the sampler requested
    """
    return _get_item(name, samplers, 'samplers')


def _get_item(name, item_list, factory_type):
    for item in item_list:
        if item.__name__ == name:
            return item
    raise ValueError('The item with the name {0} could not be found in the {1} factory.'.format(name, factory_type))
