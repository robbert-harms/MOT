__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractCLRoutine(object):

    def __init__(self, cl_environments, load_balancer, **kwargs):
        """This class serves as an abstract basis for all CL routine classes.

        Args:
            cl_environments (list of CLEnvironment): The list of CL environments using by this routine.
            load_balancer (LoadBalancingStrategy): The load balancing strategy to be used by this routine.
        """
        self._cl_environments = cl_environments
        self._load_balancer = load_balancer

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

    def _create_workers(self, worker_class, arg_list=None, kwarg_dict=None):
        """Create workers for all the CL environments in current use.

        Args:
            worker_class (class): The class to construct
            args_list (list): the arguments to pass to the constructor
            kwargs_dict (dict): the arguments to pass to the constructor
        """
        arg_list = arg_list or []
        kwarg_dict = kwarg_dict or {}
        cl_environments = self.load_balancer.get_used_cl_environments(self.cl_environments)
        return [worker_class(env, *arg_list, **kwarg_dict) for env in cl_environments]

    @classmethod
    def get_pretty_name(cls):
        """The pretty name of this routine.

        This is used to create an object of the implementing class using a factory, and is used in the logging.

        Returns:
            str: the pretty name of this routine.
        """
        return cls.__name__
