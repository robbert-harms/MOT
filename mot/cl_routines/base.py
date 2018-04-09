from mot import configuration
from mot.configuration import get_compile_flags_to_disable_in_double_precision
import numpy as np

__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CLRoutine(object):

    def __init__(self, cl_environments=None, load_balancer=None, compile_flags=None,
                 double_precision=False, **kwargs):
        """Base class for CL routines.

        Args:
            cl_environments (list of CLEnvironment): The list of CL environments using by this routine.
                If None is given we use the defaults in the current configuration.
            load_balancer (LoadBalancingStrategy): The load balancing strategy to be used by this routine.
                If None is given we use the defaults in the current configuration.
            compile_flags (dict): the list of compile flags to use during model fitting. As values use the
                flag name, as keys a boolean flag indicating if that one is active.
            double_precision (boolean): if we apply the computations in double precision or in single float precision.
                By default we go for single float precision.
        """
        self._cl_environments = cl_environments
        self._load_balancer = load_balancer
        self.compile_flags = compile_flags

        self._double_precision = double_precision
        self._mot_float_dtype = np.float32
        if self._double_precision:
            self._mot_float_dtype = np.float64

        if self._cl_environments is None:
            self._cl_environments = configuration.get_cl_environments()

        if self._load_balancer is None:
            self._load_balancer = configuration.get_load_balancer()

        if self.compile_flags is None:
            self.compile_flags = configuration.get_compile_flags(self.__class__.__name__)

    def get_cl_routine_kwargs(self):
        """Get a dictionary with the keyword arguments needed to create a similar CL routine instance.

        Returns:
            dict: a dictionary with the keyword arguments that a CLRoutine will take. This can be used to
                generate other CL routines with the same settings.
        """
        return dict(cl_environments=self.cl_environments,
                    load_balancer=self.load_balancer,
                    compile_flags=self.compile_flags,
                    double_precision=self.double_precision)

    def set_compile_flag(self, compile_flag, enable):
        """Enable or disable the given compile flag.

        Args:
            compile_flag (str): the compile flag we want to enable or disable
            enable (boolean): if we enable (True) or disable (False) this compile flag
        """
        self.compile_flags.update({compile_flag: enable})

    def get_compile_flags_list(self):
        """Get a list of the enabled and applicable compile flags.

        Returns:
            list: the list of enabled compile flags.
        """
        elements = [flag for flag, enabled in self.compile_flags.items() if enabled]

        if self._double_precision:
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

    @property
    def double_precision(self):
        return self._double_precision

    @double_precision.setter
    def double_precision(self, double_precision):
        self._double_precision = double_precision
        self._mot_float_dtype = np.float32
        if self._double_precision:
            self._mot_float_dtype = np.float64

    def _create_workers(self, worker_generating_cb):
        """Create workers for all the CL environments in current use.

        Args:
            worker_generating_cb (python function): the callback function that we use to generate the
                worker for a specific CL environment. This should accept as single argument a CL environment and
                should return a Worker instance for use in CL computations.
        """
        cl_environments = self.load_balancer.get_used_cl_environments(self.cl_environments)
        return [worker_generating_cb(env) for env in cl_environments]
