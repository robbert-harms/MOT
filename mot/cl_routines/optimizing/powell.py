import os
from pkg_resources import resource_filename
from .base import AbstractParallelOptimizer

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Powell(AbstractParallelOptimizer):

    default_patience = 2

    def __init__(self, reset_method=None, patience=None, optimizer_settings=None, patience_line_search=None, **kwargs):
        """Use the Powell method to calculate the optimum.

        Args:
            patience (int): Used to set the maximum number of iterations to patience*(number_of_parameters+1)
            reset_method (str): one of 'EXTRAPOLATED_POINT' or 'RESET_TO_IDENTITY' lower case or upper case.
            patience_line_search (int): the patience of the searching algorithm. Defaults to the
                same patience as for the Powell algorithm itself.
        """
        patience = patience or self.default_patience

        optimizer_settings = optimizer_settings or {}

        keyword_values = {}
        keyword_values['reset_method'] = reset_method
        keyword_values['patience_line_search'] = patience_line_search

        option_defaults = {'reset_method': 'EXTRAPOLATED_POINT', 'patience_line_search': patience}

        def get_value(option_name):
            value = keyword_values.get(option_name)
            if value is None:
                value = optimizer_settings.get(option_name)
            if value is None:
                value = option_defaults[option_name]
            return value

        for option in option_defaults:
            optimizer_settings.update({option: get_value(option)})

        super(Powell, self).__init__(patience=patience, optimizer_settings=optimizer_settings, **kwargs)

    def _get_optimization_function(self, model):
        params = {'NMR_PARAMS': model.get_nmr_parameters(), 'PATIENCE': self.patience}

        for option, value in self._optimizer_settings.items():
            params.update({option.upper(): value})
        params['RESET_METHOD'] = 'POWELL_RESET_METHOD_' + params['RESET_METHOD'].upper()

        powell_code = open(os.path.abspath(resource_filename('mot', 'data/opencl/powell.cl')), 'r').read()
        if params:
            powell_code = powell_code % params
        return powell_code

    def _get_optimizer_call_name(self):
        return 'powell'
