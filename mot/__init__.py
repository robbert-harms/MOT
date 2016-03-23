import logging

__author__ = 'Robbert Harms'
__date__ = '2015-01-01'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"

VERSION = '0.1.17'
VERSION_STATUS = ''

_items = VERSION.split('-')
VERSION_NUMBER_PARTS = tuple(int(i) for i in _items[0].split('.'))
if len(_items) > 1:
    VERSION_STATUS = _items[1]
__version__ = VERSION

try:
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
logging.getLogger(__name__).addHandler(NullHandler())
