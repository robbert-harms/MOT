from mot.cl_runtime_info import CLRuntimeInfo

__author__ = 'Robbert Harms'
__date__ = "2014-04-26"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CLRoutine(object):

    def __init__(self, cl_runtime_info=None):
        """Base class for CL routines.

        Args:
            cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the runtime information
        """
        self._cl_runtime_info = cl_runtime_info or CLRuntimeInfo()

    def set_cl_runtime_info(self, cl_runtime_info):
        """Update the CL runtime information.

        Args:
            cl_runtime_info (mot.cl_runtime_info.CLRuntimeInfo): the new runtime information
        """
        self._cl_runtime_info = cl_runtime_info

    # todo remove in future versions
    def _create_workers(self, worker_generating_cb):
        """Create workers for all the CL environments in current use.

        Args:
            worker_generating_cb (python function): the callback function that we use to generate the
                worker for a specific CL environment. This should accept as single argument a CL environment and
                should return a Worker instance for use in CL computations.
        """
        cl_environments = self._cl_runtime_info.get_cl_environments()
        return [worker_generating_cb(env) for env in cl_environments]
