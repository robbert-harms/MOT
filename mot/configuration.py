"""Contains the runtime configuration of MOT.

This consists of two parts, functions to get the current runtime settings and configuration actions to update these
settings. To set a new configuration, create a new :py:class:`ConfigAction` and use this within a context environment
using :py:func:`config_context`. Example:

.. code-block:: python

    from mot.configuration import RuntimeConfigurationAction, config_context

    with config_context(RuntimeConfigurationAction(...)):
        ...

"""
from contextlib import contextmanager
from copy import copy
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
    'cl_environments': CLEnvironmentFactory.smart_device_selection(),
    'load_balancer': PreferGPU(),
    'compile_flags': {
        'general': {
            '-cl-single-precision-constant': True,
            '-cl-denorms-are-zero': True,
            '-cl-mad-enable': True,
            '-cl-no-signed-zeros': True
        },
        'cl_routine_specific': {}
    },
    'ranlux': {
        'seed': 1,
        'lux_factor': 4
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

    Raises:
        ValueError: if the list of environments is empty
    """
    if not cl_environments:
        raise ValueError('The list of CL Environments is empty.')
    _config['cl_environments'] = cl_environments


def get_load_balancer():
    """Get the current load balancer to use during CL calculations.

    Returns:
        LoadBalanceStrategy: the current load balancer to use
    """
    return _config['load_balancer']


def set_load_balancer(load_balancer):
    """Set the current CL environments to the given list

    Args:
        load_balancer (LoadBalanceStrategy): the current load balancer to use
    """
    _config['load_balancer'] = load_balancer


def get_compile_flags(cl_routine_name=None):
    """Get the default compile flags to use in a CL routine.

    Args:
        cl_routine_name (str): the name of the CL routine for which we want the compile flags. If not given
            we return the default flags. If given we return the default flags updated with the routine specific flags.

    Returns:
        dict: the default list of compile flags we wish to use
    """
    flags = copy(_config['compile_flags']['general'])
    if cl_routine_name in _config['compile_flags']['cl_routine_specific']:
        flags.update(_config['compile_flags']['cl_routine_specific'][cl_routine_name])
    return flags


def get_ranlux_seed():
    """Get the default seed to use for all random number generation.

    This can be overwritten by the user when calling the script initialize_ranlux, but normally this is the
    seed that should be used.

    Returns:
        int: the seed to use during all RNG with ranlux.
    """
    return _config['ranlux']['seed']


def get_ranlux_lux_factor():
    """Get the default lux factor to use for all random number generation with ranlux.

    This can be overwritten by the user when calling the script initialize_ranlux, but normally this is the
    lux factor that should be used.

    Returns:
        int: the luxury level of the ranluxcl generator. See the ranluxcl.cl source for details.
    """
    return _config['ranlux']['lux_factor']


def set_ranlux_seed(seed):
    """Set the default seed to use for all random number generation.

    Args:
        int: the seed to use during all RNG with ranlux.
    """
    _config['ranlux']['seed'] = seed


def set_ranlux_lux_factor(lux_factor):
    """Set the default lux factor to use for all random number generation with ranlux.

    Args:
        int: the luxury level of the ranluxcl generator. See the ranluxcl.cl source for details.
    """
    _config['ranlux']['lux_factor'] = lux_factor


@contextmanager
def config_context(config_action):
    """Creates a context in which the config action is applied and unapplies the configuration after execution.

    Args:
        config_action (ConfigAction): the configuration action to use
    """
    config_action.apply()
    yield
    config_action.unapply()


class ConfigAction(object):

    def __init__(self):
        """Defines a configuration action for use in a configuration context.

        This should define an apply and unapply function that sets and unsets the configuration options.

        The applying action needs to remember the state before the application of the action.
        """

    def apply(self):
        """Apply the current action to the current runtime configuration."""

    def unapply(self):
        """Reset the current configuration to the previous state."""


class SimpleConfigAction(ConfigAction):

    def __init__(self):
        """Defines a default implementation of a configuration action.

        This simple config implements a default ``apply()`` method that saves the current state and a default
        ``unapply()`` that restores the previous state.

        For developers, it is easiest to implement ``_apply()`` such that you do not manually need to store the old
        configuraration.
        """
        super(SimpleConfigAction, self).__init__()
        self._old_config = {}

    def apply(self):
        """Apply the current action to the current runtime configuration."""
        self._old_config = {k: v for k, v in _config.items()}
        self._apply()

    def unapply(self):
        """Reset the current configuration to the previous state."""
        for key, value in self._old_config.items():
            _config[key] = value

    def _apply(self):
        """Implement this function add apply() logic after this class saves the current config."""


class RuntimeConfigurationAction(SimpleConfigAction):

    def __init__(self, cl_environments=None, load_balancer=None):
        """Updates the runtime settings.

        Args:
            cl_environments (list of CLEnvironment): the new CL environments we wish to use for future computations
            load_balancer (LoadBalanceStrategy): the load balancer to use
        """
        super(RuntimeConfigurationAction, self).__init__()
        self._cl_environments = cl_environments
        self._load_balancer = load_balancer

    def _apply(self):
        if self._cl_environments is not None:
            set_cl_environments(self._cl_environments)

        if self._load_balancer is not None:
            set_load_balancer(self._load_balancer)
