import numpy as np
from mot.utils import results_to_dict

from mot.cl_data_type import SimpleCLDataType
from mot.model_building.data_adapter import SimpleDataAdapter
from mot.model_interfaces import OptimizeModelInterface

__author__ = 'Robbert Harms'
__date__ = "2015-04-02"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Rosenbrock(OptimizeModelInterface):

    def __init__(self, n=5):
        """When optimized the parameters should all be equal to 1."""
        super(OptimizeModelInterface, self).__init__()
        self.n = n

    def double_precision(self):
        return True

    @property
    def name(self):
        return 'rosenbrock'

    def get_data(self):
        return {}

    def get_kernel_data_struct(self, device):
        return '''
            typedef struct{
                constant void* place_holder;
            } ''' + self.get_kernel_data_struct_type() + ''';
        '''

    def get_kernel_data_struct_type(self):
        return 'optimize_data'

    def get_kernel_param_names(self, device):
        return []

    def get_kernel_data_struct_initialization(self, device, variable_name, problem_id_name='gid'):
        return self.get_kernel_data_struct_type() + ' ' + variable_name + ' = {0};'

    def get_nmr_problems(self):
        return 1

    def get_model_eval_function(self, fname='evaluateModel'):
        return '''
            double ''' + fname + '''(const void* const data, const double* const x,
                                     const uint observation_index){
                double sum = 0;
                for(uint i = 0; i < ''' + str(self.n) + ''' - 1; i++){
                    sum += 100 * pown((x[i + 1] - pown(x[i], 2)), 2) + pown((x[i] - 1), 2);
                }
                return -sum;
            }
        '''

    def get_observation_return_function(self, fname='getObservation'):
        return '''
            double ''' + fname + '''(const void* const data, const uint observation_index){
                return 0;
            }
        '''

    def get_objective_function(self, fname="calculateObjective"):
        eval_fname = fname + '_evaluateModel'
        obs_fname = fname + '_getObservation'
        func = self.get_model_eval_function(eval_fname)
        func += self.get_observation_return_function(obs_fname)
        return func + '''
            double ''' + fname + '''(const void* const data, double* const x){
                return ''' + obs_fname + '''(data, 1) -''' + eval_fname + '''(data, x, 1);
            }
        '''

    def get_objective_per_observation_function(self, func_name="getObjectiveInstanceValue"):
        eval_fname = func_name + '_evaluateModel'
        obs_fname = func_name + '_getObservation'
        func = self.get_model_eval_function(eval_fname)
        func += self.get_observation_return_function(obs_fname)
        return func + '''
            mot_float_type ''' + func_name + '''(const void* const data, mot_float_type* const x,
                                                 uint observation_index){
                return ''' + obs_fname + '''(data, observation_index) -
                            ''' + eval_fname + '''(data, x, observation_index);
            }
        '''

    def get_initial_parameters(self, previous_results=None):
        params = np.ones((1, self.n)) * 3

        if isinstance(previous_results, np.ndarray):
            previous_results = results_to_dict(previous_results, self.get_free_param_names())

        if previous_results:
            for i in range(self.n):
                if i in previous_results:
                    params[0, i] = previous_results[i]
        return SimpleDataAdapter(params, SimpleCLDataType.from_string('double'),
                                 SimpleCLDataType.from_string('double')).get_opencl_data()

    def get_lower_bounds(self):
        return ['-inf'] * self.n

    def get_upper_bounds(self):
        return ['inf'] * self.n

    def get_free_param_names(self):
        return range(self.n)

    def get_nmr_inst_per_problem(self):
        return 1

    def get_nmr_estimable_parameters(self):
        return self.n


class MatlabLSQNonlinExample(OptimizeModelInterface):

    def __init__(self):
        """When optimized the parameters should be close to [0.2578, 0.2578] or something with a similar 2 norm.

        See the matlab manual page at http://nl.mathworks.com/help/optim/ug/lsqnonlin.html for more information.
        (viewed at 2015-04-02).

        """
        super(OptimizeModelInterface, self).__init__()

    def double_precision(self):
        return True

    @property
    def name(self):
        return 'matlab_lsqnonlin_example'

    def get_data(self):
        return {}

    def get_kernel_data_struct(self, device):
        return '''
            typedef struct{
                constant void* place_holder;
            } ''' + self.get_kernel_data_struct_type() + ''';
        '''

    def get_kernel_data_struct_type(self):
        return 'optimize_data'

    def get_kernel_param_names(self, device):
        return []

    def get_kernel_data_struct_initialization(self, device, variable_name, problem_id_name='gid'):
        return self.get_kernel_data_struct_type() + ' ' + variable_name + ' = {0};'

    def get_nmr_problems(self):
        return 1

    def get_model_eval_function(self, fname='evaluateModel'):
        return '''
            double ''' + fname + '''(const void* const data, const double* const x,
                                     const uint k){
                return -(2 + 2 * (k+1) - exp((k+1) * x[0]) - exp((k+1) * x[1]));
            }
        '''

    def get_observation_return_function(self, fname='getObservation'):
        return '''
            double ''' + fname + '''(const void* const data, const uint observation_index){
                return 0;
            }
        '''

    def get_objective_function(self, fname="calculateObjective"):
        eval_fname = fname + '_evaluateModel'
        obs_fname = fname + '_getObservation'
        func = self.get_model_eval_function(eval_fname)
        func += self.get_observation_return_function(obs_fname)
        return func + '''
            double ''' + fname + '''(const void* const data, double* const x){
                double sum = 0;
                for(uint i = 0; i < 10; i++){
                    sum += ''' + obs_fname + '''(data, i) - ''' + eval_fname + '''(data, x, i);
                }
                return sum;
            }
        '''

    def get_objective_per_observation_function(self, func_name="getObjectiveInstanceValue"):
        eval_fname = func_name + '_evaluateModel'
        obs_fname = func_name + '_getObservation'
        func = self.get_model_eval_function(eval_fname)
        func += self.get_observation_return_function(obs_fname)
        return func + '''
            mot_float_type ''' + func_name + '''(const void* const data, mot_float_type* const x,
                                                 uint observation_index){
                return ''' + obs_fname + '''(data, observation_index) -
                            ''' + eval_fname + '''(data, x, observation_index);
            }
        '''

    def get_initial_parameters(self, previous_results=None):
        params = np.array([[0.3, 0.4]])

        if isinstance(previous_results, np.ndarray):
            previous_results = results_to_dict(previous_results, self.get_free_param_names())

        if previous_results:
            for i in range(2):
                if i in previous_results:
                    params[0, i] = previous_results[i]
        return SimpleDataAdapter(params, SimpleCLDataType.from_string('double'),
                                 SimpleCLDataType.from_string('double')).get_opencl_data()

    def get_lower_bounds(self):
        return [0, 0]

    def get_upper_bounds(self):
        return ['inf', 'inf']

    def get_free_param_names(self):
        return [0, 1]

    def get_nmr_inst_per_problem(self):
        return 10

    def get_nmr_estimable_parameters(self):
        return 2
