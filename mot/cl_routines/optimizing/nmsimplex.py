from mot.library_functions import LibNMSimplex
from ...cl_routines.optimizing.base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class NMSimplex(AbstractParallelOptimizer):

    default_patience = 200

    def __init__(self, scale=None, alpha=None, beta=None, gamma=None, delta=None, adaptive_scales=None,
                 patience=None, optimizer_settings=None, **kwargs):
        """Use the Nelder-Mead simplex method to calculate the optimimum.

        Args:
            patience (int): Used to set the maximum number of iterations to patience*(number_of_parameters+1)
            scale (double): the scale of the initial simplex, default 1.0
            alpha (double): reflection coefficient, default 1.0
            beta (double): contraction coefficient, default 0.5
            gamma (double); expansion coefficient, default 2.0
            delta (double); shrinkage coefficient, default 0.5
            adaptive_scales (boolean): if set to True we use adaptive scales instead of the default scale values.
                This sets the scales to:

                .. code-block:: python

                    n = model.get_nmr_estimable_parameters()

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

        option_defaults = {'alpha': 1.0, 'beta': 0.5, 'gamma': 2.0, 'delta': 0.5, 'scale': 1.0,
                           'adaptive_scales': True}

        def get_value(option_name):
            value = keyword_values.get(option_name)
            if value is None:
                value = optimizer_settings.get(option_name)
            if value is None:
                value = option_defaults[option_name]
            return value

        for option in option_defaults:
            optimizer_settings.update({option: get_value(option)})

        super(NMSimplex, self).__init__(patience=patience, optimizer_settings=optimizer_settings, **kwargs)

    def minimize(self, model, init_params=None):
        if self._optimizer_settings.get('adaptive_scales', True):
            nmr_params = model.get_nmr_estimable_parameters()
            self._optimizer_settings.update(
                {'alpha': 1,
                 'beta': 0.75 - 1.0 / (2 * nmr_params),
                 'gamma': 1 + 2.0 / nmr_params,
                 'delta': 1 - 1.0 / nmr_params}
                )
        return super(NMSimplex, self).minimize(model, init_params=init_params)

    def _get_worker_generator(self, *args):
        return lambda cl_environment: NMSimplexWorker(cl_environment, *args)


class NMSimplexWorker(AbstractParallelOptimizerWorker):

    def _get_optimization_function(self):
        params = {'NMR_PARAMS': self._nmr_params, 'PATIENCE': self._parent_optimizer.patience}

        for option, value in self._optimizer_settings.items():
            if option == 'scale':
                params['INITIAL_SIMPLEX_SCALES'] = '{' + ', '.join([str(value)] * self._nmr_params) + '}'
            else:
                params.update({option.upper(): value})

        lib_nmsimplex = LibNMSimplex(evaluate_fname='evaluate')
        body = lib_nmsimplex.get_cl_code()

        body += '''
            int nmsimplex(mot_float_type* const model_parameters, const void* const data){
                mot_float_type initial_simplex_scale[%(NMR_PARAMS)r] = %(INITIAL_SIMPLEX_SCALES)s;
                mot_float_type fdiff;
                mot_float_type psi = 0;
                mot_float_type nmsimplex_scratch[%(NMR_PARAMS)r * 2 + (%(NMR_PARAMS)r + 1) * (%(NMR_PARAMS)r + 1)];

                return lib_nmsimplex(%(NMR_PARAMS)r, model_parameters, data, initial_simplex_scale,
                                     &fdiff, psi, (int)(%(PATIENCE)r * (%(NMR_PARAMS)r+1)),
                                     %(ALPHA)r, %(BETA)r, %(GAMMA)r, %(DELTA)r,
                                     nmsimplex_scratch);
            }
        ''' % params

        return body

    def _get_optimizer_call_name(self):
        return 'nmsimplex'
