import numpy as np
from mot import configuration
from mot.configuration import get_compile_flags_to_disable_in_double_precision

__author__ = 'Robbert Harms'
__date__ = '2018-04-23'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class CLRuntimeInfo(object):

    def __init__(self, cl_environments=None, load_balancer=None, compile_flags=None, double_precision=None):
        """All information necessary for applying operations using OpenCL.

        Args:
            cl_environments (list of mot.cl_environments.CLEnvironment): The list of CL environments used by
                this routine. If None is given we use the defaults in the current configuration.
            load_balancer (LoadBalancingStrategy): The load balancing strategy to be used by this routine.
                If None is given we use the defaults in the current configuration.
            compile_flags (list): the list of compile flags to use during analysis.
            double_precision (boolean): if we apply the computations in double precision or in single float precision.
                By default we go for single float precision.
        """
        self._cl_environments = cl_environments
        self._load_balancer = load_balancer
        self._compile_flags = compile_flags
        self._double_precision = double_precision

        if self._cl_environments is None:
            self._cl_environments = configuration.get_cl_environments()

        if self._load_balancer is None:
            self._load_balancer = configuration.get_load_balancer()

        if self._compile_flags is None:
            self._compile_flags = configuration.get_compile_flags()

        if self._double_precision is None:
            self._double_precision = configuration.use_double_precision()

    @property
    def cl_environments(self):
        return self._cl_environments

    @property
    def mot_float_dtype(self):
        if self._double_precision:
            return np.float64
        return np.float32

    @property
    def load_balancer(self):
        return self._load_balancer

    @property
    def double_precision(self):
        return self._double_precision

    def get_compile_flags(self):
        """Get a list of the applicable compile flags.

        Returns:
            list: the list of enabled compile flags.
        """
        elements = list(self._compile_flags)
        if self._double_precision:
            elements_to_remove = get_compile_flags_to_disable_in_double_precision()
            elements = list(filter(lambda e: e not in elements_to_remove, elements))
        return elements

    def get_cl_environments(self):
        """Get a list of the cl environments to use.

        This returns only the CL environments that will be used by the load balancer.

        Returns:
            list of mot.cl_environments.CLEnvironment: a list of CL environments to use
        """
        return self._load_balancer.get_used_cl_environments(self._cl_environments)
