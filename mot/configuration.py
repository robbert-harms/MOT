from contextlib import contextmanager
from .cl_environments import CLEnvironmentFactory
from .load_balance_strategies import PreferGPU

__author__ = 'Robbert Harms'
__date__ = "2015-07-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


"""The runtime configuration, this can be overwritten at run time.

For any of the AbstractCLRoutines it holds that if no suitable defaults are given we use the ones provided by this
module. This entire module acts as a singleton containing the current runtime configuration.
"""
_config = {
    'cl_environments': CLEnvironmentFactory.all_devices(),
    'load_balancer': PreferGPU(),
    'compile_flags': {
        '-cl-single-precision-constant': True,
        '-cl-denorms-are-zero': True,
        '-cl-mad-enable': True,
        '-cl-no-signed-zeros': True
    }
}

"""If we ignore kernel warnings or not"""
ignore_kernel_compile_warnings = True


def get_cl_environments():
    """Get the current CL environment to use during CL calculations.

    Returns:
        list of CLEnvironment: the current list of CL environments.
    """
    return _config['cl_environments']


def set_cl_environments(cl_environments):
    """Set the current CL environments to the given list

    Args:
        cl_environments (list of CLEnvironment): the new list of CL environments.
    """
    _config['cl_environments'] = cl_environments


def get_load_balancer():
    """Get the current load balancer to use during CL calculations.

    Returns:
        LoadBalancer: the current load balancer to use
    """
    return _config['load_balancer']


def set_load_balancer(load_balancer):
    """Set the current CL environments to the given list

    Args:
        load_balancer (LoadBalancer): the current load balancer to use
    """
    _config['load_balancer'] = load_balancer


def get_compile_flags():
    """Get the default compile flags to use in a CL routine.

    Returns:
        dict: the default list of compile flags we wish to use
    """
    return _config['compile_flags']


@contextmanager
def config_context(config_action):
    config_action.apply()
    yield
    config_action.unapply()


class ConfigAction(object):

    def __init__(self):
        """Defines a configuration action for the use in a configuration context.

        This should define an apply and an unapply function that sets and unsets the given configuration options.

        The applying action needs to remember the state before applying the action.
        """
        self._old_config = {}

    def apply(self):
        """Apply the current action to the current runtime configuration."""
        self._old_config = {k: v for k, v in _config.items()}

    def unapply(self):
        """Reset the current configuration to the previous state."""
        for key, value in self._old_config.items():
            _config[key] = value


class RuntimeConfigurationAction(ConfigAction):

    def __init__(self, cl_environments=None, load_balancer=None):
        super(RuntimeConfigurationAction, self).__init__()
        self._cl_environments = cl_environments
        self._load_balancer = load_balancer

    def apply(self):
        super(RuntimeConfigurationAction, self).apply()

        if self._cl_environments is not None:
            set_cl_environments(self._cl_environments)

        if self._load_balancer is not None:
            set_load_balancer(self._load_balancer)
