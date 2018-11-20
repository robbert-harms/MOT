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
import numpy as np
from .lib.cl_environments import CLEnvironmentFactory

__author__ = 'Robbert Harms'
__date__ = "2015-07-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


"""The runtime configuration, this can be overwritten at run time.

For any of the AbstractCLRoutines it holds that if no suitable defaults are given we use the ones provided by this
module. This entire module acts as a singleton containing the current runtime configuration.
"""
_config = {
    'cl_environments': CLEnvironmentFactory.smart_device_selection(preferred_device_type='GPU'),
    'compile_flags': ['-cl-denorms-are-zero', '-cl-mad-enable', '-cl-no-signed-zeros'],
    'double_precision': False
}


def get_cl_environments():
    """Get the current CL environment to use during CL calculations.

    Returns:
        list of CLEnvironment: the current list of CL environments.
    """
    return _config['cl_environments']


def set_cl_environments(cl_environments):
    """Set the current CL environments to the given list

    Please note that this will change the global configuration, i.e. this is a persistent change. If you do not want
    a persistent state change, consider using :func:`~mot.configuration.config_context` instead.

    Args:
        cl_environments (list of CLEnvironment): the new list of CL environments.

    Raises:
        ValueError: if the list of environments is empty
    """
    if not cl_environments:
        raise ValueError('The list of CL Environments is empty.')
    _config['cl_environments'] = cl_environments


def get_compile_flags():
    """Get the default compile flags to use in a CL routine.

    Returns:
        list: the default list of compile flags we wish to use
    """
    return list(_config['compile_flags'])


def set_compile_flags(compile_flags):
    """Set the current compile flags.

    Args:
        compile_flags (list): the new list of compile flags
    """
    _config['compile_flags'] = compile_flags


def set_default_proposal_update(proposal_update):
    """Set the default proposal update function to use in sample.

    Please note that this will change the global configuration, i.e. this is a persistent change. If you do not want
    a persistent state change, consider using :func:`~mot.configuration.config_context` instead.

    Args:
        mot.model_building.parameter_functions.proposal_updates.ProposalUpdate: the new proposal update function
            to use by default if no specific one is provided.
    """
    _config['default_proposal_update'] = proposal_update


def use_double_precision():
    """Check if we run the computations in default precision or not.

    Returns:
        boolean: if we run the computations in double precision or not
    """
    return _config['double_precision']


def set_use_double_precision(double_precision):
    """Set the default use of double precision.

    Returns:
        boolean: if we use double precision by default or not
    """
    _config['double_precision'] = double_precision


@contextmanager
def config_context(config_action):
    """Creates a context in which the config action is applied and unapplies the configuration after execution.

    Args:
        config_action (ConfigAction): the configuration action to use
    """
    config_action.apply()
    yield
    config_action.unapply()


class ConfigAction:

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
        super().__init__()
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


class CLRuntimeAction(SimpleConfigAction):

    def __init__(self, cl_runtime_info):
        """Set the current configuration to use the information in the given configuration action.

        Args:
            cl_runtime_info (CLRuntimeInfo): the runtime info with the configuration options
        """
        super().__init__()
        self._cl_runtime_info = cl_runtime_info

    def _apply(self):
        set_cl_environments(self._cl_runtime_info.cl_environments)
        set_compile_flags(self._cl_runtime_info._compile_flags)
        set_use_double_precision(self._cl_runtime_info.double_precision)


class RuntimeConfigurationAction(SimpleConfigAction):

    def __init__(self, cl_environments=None, compile_flags=None, double_precision=None):
        """Updates the runtime settings.

        Args:
            cl_environments (list of CLEnvironment): the new CL environments we wish to use for future computations
            compile_flags (list): the list of compile flags to use during analysis.
            double_precision (boolean): if we compute in double precision or not
        """
        super().__init__()
        self._cl_environments = cl_environments
        self._compile_flags = compile_flags
        self._double_precision = double_precision

    def _apply(self):
        if self._cl_environments is not None:
            set_cl_environments(self._cl_environments)

        if self._compile_flags is not None:
            set_compile_flags(self._compile_flags)

        if self._double_precision is not None:
            set_use_double_precision(self._double_precision)


class VoidConfigurationAction(ConfigAction):

    def __init__(self):
        """Does nothing, useful as a default config action.
        """
        super().__init__()


class CLRuntimeInfo:

    def __init__(self, cl_environments=None, compile_flags=None, double_precision=None):
        """All information necessary for applying operations using OpenCL.

        Args:
            cl_environments (list of mot.lib.cl_environments.CLEnvironment): The list of CL environments used by
                this routine. If None is given we use the defaults in the current configuration.
            compile_flags (list): the list of compile flags to use during analysis.
            double_precision (boolean): if we apply the computations in double precision or in single float precision.
                By default we go for single float precision.
        """
        self._cl_environments = cl_environments
        self._compile_flags = compile_flags
        self._double_precision = double_precision

        if self._cl_environments is None:
            self._cl_environments = get_cl_environments()

        if self._compile_flags is None:
            self._compile_flags = get_compile_flags()

        if self._double_precision is None:
            self._double_precision = use_double_precision()

    @property
    def cl_environments(self):
        return self._cl_environments

    @property
    def mot_float_dtype(self):
        if self._double_precision:
            return np.float64
        return np.float32

    @property
    def double_precision(self):
        return self._double_precision

    @property
    def compile_flags(self):
        """Get all defined compile flags."""
        return self._compile_flags
