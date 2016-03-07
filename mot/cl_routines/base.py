__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class AbstractCLRoutine(object):

    def __init__(self, cl_environments, load_balancer, compile_flags=None, **kwargs):
        """This class serves as an abstract basis for all CL routine classes.

        Args:
            cl_environments (list of CLEnvironment): The list of CL environments using by this routine.
            load_balancer (LoadBalancingStrategy): The load balancing strategy to be used by this routine.
            compile_flags (dict): the list of compile flags we want to enable or disable. The keys (str)
                should be the compile flag, the value (boolean) specifies enable or disable.
        """
        self._cl_environments = cl_environments
        self._load_balancer = load_balancer
        self.compile_flags = {
            '-cl-single-precision-constant': True,
            '-cl-denorms-are-zero': True,
            '-cl-strict-aliasing': True,
            '-cl-mad-enable': True,
            '-cl-no-signed-zeros': True
        }

        if compile_flags:
            self.compile_flags.update(compile_flags)

    def set_compile_flag(self, compile_flag, enable):
        """Enable or disable the given compile flag.

        Args:
            compile_flag (str): the compile flag we want to enable or disable
            enable (boolean): if we enable (True) or disable (False) this compile flag
        """
        self.compile_flags.update({compile_flag: enable})

    def get_compile_flags_list(self):
        """Get a list of the enabled compile flags.

        Returns:
            list: the list of enabled compile flags.
        """
        return [flag for flag, enabled in self.compile_flags.items() if enabled]

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

    @classmethod
    def get_pretty_name(cls):
        """The pretty name of this routine.

        This is used to create an object of the implementing class using a factory, and is used in the logging.

        Returns:
            str: the pretty name of this routine.
        """
        return cls.__name__
