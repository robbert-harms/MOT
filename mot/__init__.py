import logging
from .__version__ import VERSION, VERSION_STATUS, __version__

from mot.cl_routines.sampling.metropolis_hastings import MetropolisHastings
from mot.cl_routines.optimizing.levenberg_marquardt import LevenbergMarquardt
from mot.cl_routines.optimizing.powell import Powell
from mot.cl_routines.optimizing.nmsimplex import NMSimplex


try:
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
logging.getLogger(__name__).addHandler(NullHandler())


def smart_device_selection():
    """Get a list of device environments that is suitable for use in MOT.

    Returns:
        list of CLEnvironment: List with the CL device environments.
    """
    from mot.cl_environments import CLEnvironmentFactory
    return CLEnvironmentFactory.smart_device_selection()
