from mot import configuration
from mot.configuration import get_compile_flags_to_disable_in_double_precision

__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CLRoutine(object):

    def __init__(self, cl_environments=None, load_balancer=None, compile_flags=None, **kwargs):
        """Base class for CL routines. Im

        Args:
            cl_environments (list of CLEnvironment): The list of CL environments using by this routine.
                If None is given we use the defaults in the current configuration.
            load_balancer (LoadBalancingStrategy): The load balancing strategy to be used by this routine.
                If None is given we use the defaults in the current configuration.
            compile_flags (dict): the list of compile flags to use during model fitting. As values use the
                flag name, as keys a boolean flag indicating if that one is active.
        """
        self._cl_environments = cl_environments
        self._load_balancer = load_balancer
        self.compile_flags = compile_flags

        if self._cl_environments is None:
            self._cl_environments = configuration.get_cl_environments()

        if self._load_balancer is None:
            self._load_balancer = configuration.get_load_balancer()

        if self.compile_flags is None:
            self.compile_flags = configuration.get_compile_flags(self.__class__.__name__)

    def set_compile_flag(self, compile_flag, enable):
        """Enable or disable the given compile flag.

        Args:
            compile_flag (str): the compile flag we want to enable or disable
            enable (boolean): if we enable (True) or disable (False) this compile flag
        """
        self.compile_flags.update({compile_flag: enable})

    def get_compile_flags_list(self, double_precision=True):
        """Get a list of the enabled compile flags.

        Args:
            double_precision (boolean): if this is set to True we remove some of the Flags that are only applicable
                when running in float mode. More specifically, this will set cl-single-precision-constant to False.
                Set this to False to disable this behaviour and use the flags as specified in the config.

        Returns:
            list: the list of enabled compile flags.
        """
        elements = [flag for flag, enabled in self.compile_flags.items() if enabled]

        if double_precision:
            elements_to_remove = get_compile_flags_to_disable_in_double_precision()
            elements = list(filter(lambda e: e not in elements_to_remove, elements))

        return elements

    @property
    def cl_environments(self):
        return self._cl_environments

    @cl_environments.setter
    def cl_environments(self, cl_environments):
        if cl_environments is not None:
            self._cl_environments = cl_environments

    @property
    def load_balancer(self):
        return self._load_balancer

    @load_balancer.setter
    def load_balancer(self, load_balancer):
        self._load_balancer = load_balancer

    def _create_workers(self, worker_generating_cb):
        """Create workers for all the CL environments in current use.

        Args:
            worker_generating_cb (python function): the callback function that we use to generate the
                worker for a specific CL environment. This should accept as single argument a CL environment and
                should return a Worker instance for use in CL computations.
        """
        cl_environments = self.load_balancer.get_used_cl_environments(self.cl_environments)
        return [worker_generating_cb(env) for env in cl_environments]
