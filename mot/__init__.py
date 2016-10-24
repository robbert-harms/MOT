import logging
from .__version__ import VERSION, VERSION_STATUS, __version__

__author__ = 'Robbert Harms'
__date__ = '2015-01-01'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"

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
