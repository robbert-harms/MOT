import math
from .base import ModelFunction, FreeParameter
from .parameter_functions.transformations import CosSqrClampTransform
from .base import CLDataType


__author__ = 'Robbert Harms'
__date__ = "2014-08-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class EvaluationModel(ModelFunction):

    def __init__(self, name, cl_function_name, parameter_list, dependency_list=()):
        """The evaluation model is the model under which you evaluate the estimated results against the data."""
        super(EvaluationModel, self).__init__(name, cl_function_name, parameter_list, dependency_list=dependency_list)

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        """Get the cl code for the objective function under the given noise model.

        Args:
            fname (str): the name of the resulting function
            inst_per_problem (int): the number of instances per problem
            eval_fname (str): the name of the function that can be called to get the evaluation, its signature is:
                model_float <fname>(const optimize_data* data, const model_float* x, const int observation_index);
            obs_fname (str): the name of the function that can be called for the observed data, its signature is:
                model_float <fname>(const optimize_data* data, const int observation_index);
            param_listing (str): the parameter listings for the parameters of the noise model

        Returns:
            The objective function under this noise model, its signature is:
                model_float <fname>(const optimize_data* const data, model_float* const x);
        """

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        """Get the cl code for the log likelihood function under the given noise model.

        Args:
            fname (str): the name of the resulting function
            inst_per_problem (int): the number of instances per problem
            eval_fname (str): the name of the function that can be called to get the evaluation, its signature is:
                model_float <fname>(const optimize_data* data, const model_float* x, const int observation_index);
            obs_fname (str): the name of the function that can be called for the observed data, its signature is:
                model_float <fname>(const optimize_data* data, const int observation_index);
            param_listing (str): the parameter listings for the parameters of the noise model

        Returns:
            the objective function under this noise model, its signature is:
                double <fname>(const optimize_data* const data, model_float* const x);
        """


class SumOfSquares(EvaluationModel):

    def __init__(self):
        """Evaluates the distance between the estimated signal and the data using the sum of squared distance."""
        super(EvaluationModel, self).__init__('SumOfSquaresNoise', 'sumOfSquaresNoise', (), ())

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            model_float ''' + fname + '''(const optimize_data* const data, model_float* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) - ''' + eval_fname + '''(data, x, i), 2);
                }
                return sum;
            }
        '''

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            double ''' + fname + '''(const optimize_data* const data, const model_float* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    double += pown(''' + obs_fname + '''(data, i) - ''' + eval_fname + '''(data, x, i), 2);
                }
                return - sum;
            }
        '''


class GaussianEvaluationModel(EvaluationModel):

    def __init__(self):
        """Evaluates the distance between the estimated signal and the data using the Gaussian evaluation."""
        super(EvaluationModel, self).__init__(
            'GaussianNoise',
            'gaussianNoiseModel',
            (FreeParameter(CLDataType.from_string('model_float'), 'sigma', False, math.sqrt(0.5), math.sqrt(0.5), 5,
                           parameter_transform=CosSqrClampTransform()),), ())

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            model_float ''' + fname + '''(const optimize_data* const data, model_float* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) - ''' + eval_fname + '''(data, x, i), 2);
                }
                return (model_float) (sum / (2 * pown(GaussianNoise_sigma, 2)));
            }
        '''

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            double ''' + fname + '''(const optimize_data* const data, const model_float* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) - ''' + eval_fname + '''(data, x, i), 2);
                }

                return - sum / (2 * pown(GaussianNoise_sigma, 2));
            }
        '''


class OffsetGaussianEvaluationModel(EvaluationModel):

    def __init__(self):
        """Evaluates the distance between the estimated signal and the data using the Offset Gaussian evaluation."""
        super(EvaluationModel, self).__init__(
            'OffsetGaussianNoise',
            'offsetGaussianNoiseModel',
            (FreeParameter(CLDataType.from_string('model_float'), 'sigma', False, math.sqrt(0.5), math.sqrt(0.5), 5,
                           parameter_transform=CosSqrClampTransform()),), ())

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            model_float ''' + fname + '''(const optimize_data* const data, model_float* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) -
                                    sqrt(pown(''' + eval_fname + '''(data, x, i), 2) +
                                                2 * pown(OffsetGaussianNoise_sigma, 2)), 2);
                }
                return (model_float) (sum / (2 * pown(OffsetGaussianNoise_sigma, 2)));
            }
        '''

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            double ''' + fname + '''(const optimize_data* const data, const model_float* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) -
                                    sqrt(pown(''' + eval_fname + '''(data, x, i), 2) +
                                                2 * pown(OffsetGaussianNoise_sigma, 2)), 2);
                }
                return - sum / (2 * pown(OffsetGaussianNoise_sigma, 2));
            }
        '''

