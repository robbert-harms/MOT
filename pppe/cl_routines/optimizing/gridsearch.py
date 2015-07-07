from ...cl_routines.optimizing.base import AbstractParallelOptimizer, AbstractParallelOptimizerWorker

__author__ = 'Robbert Harms'
__date__ = "2014-02-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class GridSearch(AbstractParallelOptimizer):

    default_patience = 250

    def __init__(self, cl_environments=None, load_balancer=None, use_param_codec=True, patience=None):
        patience = patience or self.default_patience
        super(GridSearch, self).__init__(cl_environments, load_balancer, use_param_codec, patience=patience)
        self._automatic_apply_codec = False

    def _get_worker_class(self):
        return GridSearchWorker


class GridSearchWorker(AbstractParallelOptimizerWorker):

    def _get_optimizer_cl_code(self):
        param_codec = self._model.get_parameter_codec()
        nmr_params = self._nmr_params
        lower_bounds = self._model.get_lower_bounds()
        upper_bounds = self._model.get_upper_bounds()

        kernel_source = ''
        if param_codec:
            kernel_source += param_codec.get_cl_encode_function('encodeParameters') + "\n"
            kernel_source += param_codec.get_cl_decode_function('decodeParameters') + "\n"

        kernel_source += '''
            void grid_search(double * const x, const void* const data){
                double x_optimal[''' + repr(nmr_params) + '''];
                for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                    x_optimal[i] = x[i];
                }

                ''' + ('encodeParameters(x); decodeParameters(x);' if self._use_param_codec else '') + '''

                double lowest_error = calculateObjective((optimize_data*)data, x);
                double error = 0.0;

                ''' + self._get_cl_test_loops(lower_bounds, upper_bounds, param_codec, nmr_params) + '''

                for(int i = 0; i < ''' + repr(nmr_params) + '''; i++){
                    x[i] = x_optimal[i];
                }
            }
        '''
        return kernel_source

    def _get_optimizer_call_name(self):
        return 'grid_search'

    def _get_cl_test_loops(self, lower_bounds, upper_bounds, codec, nmr_params):
        loops = ''

        evals_per_parameter = int(round((self._parent_optimizer.patience * nmr_params) ** (1.0/len(lower_bounds))))

        for i in range(len(lower_bounds)):
            loops += 'int i_' + repr(i) + '; ' + "\n"

        for i in range(len(lower_bounds)):
            loops += 'for(i_{0} = 0; i_{0} < {1}; i_{0}++){{' \
                     '      x[{0}] = {2} + i_{0} * {3};'.format(i, evals_per_parameter, lower_bounds[i],
                                                                (upper_bounds[i] - lower_bounds[i]) /
                                                                evals_per_parameter)
            loops += "\n"

        if codec is not None:
            loops += '''
                        encodeParameters(x);
                        decodeParameters(x);
            '''
        loops += """
                        error = calculateObjective((optimize_data*)data, x);
                        if(error < lowest_error){
                            lowest_error = error;"""

        loops += "\n" + "\t"*6
        for i in range(len(lower_bounds)):
            loops += "\t" + 'x_optimal[' + repr(i) + '] = x[' + repr(i) + "];\n" + "\t"*6

        loops += '}'

        for i in range(len(lower_bounds)):
            loops += """
                    }"""
        return loops