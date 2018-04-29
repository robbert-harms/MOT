import os
from pkg_resources import resource_filename
from mot.library_functions import LibNMSimplex
from ...cl_routines.optimizing.base import AbstractParallelOptimizer

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SBPlex(AbstractParallelOptimizer):

    default_patience = 10

    def __init__(self, scale=None, alpha=None, beta=None, gamma=None, delta=None, psi=None, omega=None,
                 min_subspace_length=None, max_subspace_length=None, adaptive_scales=None,
                 patience_nmsimplex=None,
                 patience=None, optimizer_settings=None, **kwargs):
        """Variation on the Nelder-Mead Simplex method by Thomas H. Rowan.

        This method uses NMSimplex to search subspace regions for the minimum. See Rowan's thesis titled
        "Functional Stability analysis of numerical algorithms" for more details.

        Args:
            patience (int): Used to set the maximum number of iterations to patience*(number_of_parameters+1)
            patience_nmsimplex (int): The maximum patience for each subspace search.
                For each subspace search we set the number of iterations to patience*(number_of_parameters_subspace+1)
            scale (double): the scale of the initial simplex, default 1.0
            alpha (double): reflection coefficient, default 1.0
            beta (double): contraction coefficient, default 0.5
            gamma (double); expansion coefficient, default 2.0
            delta (double); shrinkage coefficient, default 0.5
            psi (double): subplex specific, simplex reduction coefficient, default 0.001.
            omega (double): subplex specific, scaling reduction coefficient, default 0.01
            min_subspace_length (int): the minimum subspace length, defaults to min(2, n).
                This should hold: (1 <= min_s_d <= max_s_d <= n and min_s_d*ceil(n/nsmax_s_dmax) <= n)
            max_subspace_length (int): the maximum subspace length, defaults to min(5, n).
                This should hold: (1 <= min_s_d <= max_s_d <= n and min_s_d*ceil(n/max_s_d) <= n)

            adaptive_scales (boolean): if set to True we use adaptive scales instead of the default scale values.
                This sets the scales to:

                .. code-block:: python

                    n = model.get_nmr_parameters()

                    alpha = 1
                    beta  = 0.75 - 1.0 / (2 * n)
                    gamma = 1 + 2.0 / n
                    delta = 1 - 1.0 / n

                Following the paper:

                * Gao F, Han L. Implementing the Nelder-Mead simplex algorithm with adaptive parameters.
                  Comput Optim Appl. 2012;51(1):259-277. doi:10.1007/s10589-010-9329-3.

            optimizer_settings (dict): the optimization settings, you can use the following:
                scale, alpha, beta, gamma, delta, adaptive_scales

                The scales should satisfy the following constraints:

                .. code-block:: python

                    alpha > 0
                    0 < beta < 1
                    gamma > 1
                    gamma > alpha
                    0 < delta < 1
        """
        patience = patience or self.default_patience

        optimizer_settings = optimizer_settings or {}

        keyword_values = {}
        keyword_values['scale'] = scale
        keyword_values['alpha'] = alpha
        keyword_values['beta'] = beta
        keyword_values['gamma'] = gamma
        keyword_values['delta'] = delta
        keyword_values['adaptive_scales'] = adaptive_scales
        keyword_values['psi'] = psi
        keyword_values['omega'] = omega
        keyword_values['min_subspace_length'] = min_subspace_length
        keyword_values['max_subspace_length'] = max_subspace_length
        keyword_values['patience_nmsimplex'] = patience_nmsimplex

        option_defaults = {'alpha': 1.0, 'beta': 0.5, 'gamma': 2.0, 'delta': 0.5, 'scale': 1.0,
                           'adaptive_scales': True, 'psi': 0.001, 'omega': 0.01, 'min_subspace_length': 'auto',
                           'max_subspace_length': 'auto', 'patience_nmsimplex': 100}

        def get_value(option_name):
            value = keyword_values.get(option_name)
            if value is None:
                value = optimizer_settings.get(option_name)
            if value is None:
                value = option_defaults[option_name]
            return value

        for option in option_defaults:
            optimizer_settings.update({option: get_value(option)})

        super(SBPlex, self).__init__(patience=patience, optimizer_settings=optimizer_settings, **kwargs)

    def _get_optimization_function(self, model):
        nmr_params = model.get_nmr_parameters()
        params = {'NMR_PARAMS': nmr_params, 'PATIENCE': self.patience,
                  'ADAPTIVE_SCALES': int(self.optimizer_settings.get('adaptive_scales', True))}

        for option, value in self._optimizer_settings.items():
            if option == 'scale':
                params['INITIAL_SIMPLEX_SCALES'] = '{' + ', '.join([str(value)] * nmr_params) + '}'
            else:
                params.update({option.upper(): value})

        if self._optimizer_settings.get('min_subspace_length', 'auto') == 'auto':
            params.update({'MIN_SUBSPACE_LENGTH': min(2, nmr_params)})

        if self._optimizer_settings.get('max_subspace_length', 'auto') == 'auto':
            params.update({'MAX_SUBSPACE_LENGTH': min(5, nmr_params)})

        sbplex = open(os.path.abspath(resource_filename('mot', 'data/opencl/sbplex.cl')), 'r').read()
        if params:
            sbplex = sbplex % params

        body = sbplex
        body += LibNMSimplex(evaluate_fname='subspace_evaluate').get_cl_code()

        return body

    def _get_optimizer_call_name(self):
        return 'sbplex'
