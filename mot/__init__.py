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
