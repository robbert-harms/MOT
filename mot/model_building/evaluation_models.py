from mot.base import ModelFunction, FreeParameter
from mot.model_building.parameter_functions.transformations import CosSqrClampTransform
from mot.base import DataType


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
                MOT_FLOAT_TYPE <fname>(const optimize_data* data, const MOT_FLOAT_TYPE* x, const int observation_index);
            obs_fname (str): the name of the function that can be called for the observed data, its signature is:
                MOT_FLOAT_TYPE <fname>(const optimize_data* data, const int observation_index);
            param_listing (str): the parameter listings for the parameters of the noise model

        Returns:
            The objective function under this noise model, its signature is:
                MOT_FLOAT_TYPE <fname>(const optimize_data* const data, MOT_FLOAT_TYPE* const x);
        """

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        """Get the cl code for the log likelihood function under the given noise model.

        Args:
            fname (str): the name of the resulting function
            inst_per_problem (int): the number of instances per problem
            eval_fname (str): the name of the function that can be called to get the evaluation, its signature is:
                MOT_FLOAT_TYPE <fname>(const optimize_data* data, const MOT_FLOAT_TYPE* x, const int observation_index);
            obs_fname (str): the name of the function that can be called for the observed data, its signature is:
                MOT_FLOAT_TYPE <fname>(const optimize_data* data, const int observation_index);
            param_listing (str): the parameter listings for the parameters of the noise model

        Returns:
            the objective function under this noise model, its signature is:
                double <fname>(const optimize_data* const data, MOT_FLOAT_TYPE* const x);
        """

    def set_noise_level_std(self, noise_std):
        """Set the estimate of the noise level standard deviation.

        We put this here as a method to make the method work with object oriented polymorphism. That is, not
        all noiselevels use the same parameter for the noise standard deviation.

        This method makes no assumptions about the state of the parameters affected, they can be fixed or not. This
        function should just set the right value.

        Args:
            noise_std (double): the noise standard deviation
        """


class SumOfSquares(EvaluationModel):

    def __init__(self):
        """Evaluates the distance between the estimated signal and the data using the sum of squared distance.

        This is implemented as:
        sum((observation - evaluation)^2)
        """
        super(EvaluationModel, self).__init__('SumOfSquaresNoise', 'sumOfSquaresNoise', (), ())

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            MOT_FLOAT_TYPE ''' + fname + '''(const optimize_data* const data, MOT_FLOAT_TYPE* const x){
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
            double ''' + fname + '''(const optimize_data* const data, const MOT_FLOAT_TYPE* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) - ''' + eval_fname + '''(data, x, i), 2);
                }
                return - sum;
            }
        '''


class GaussianEvaluationModel(EvaluationModel):

    def __init__(self):
        """Evaluates the distance between the estimated signal and the data using the Gaussian evaluation.

        The offset gaussian noise is implemented as:

        sum((observation - evaluation)^2) / 2 * sigma^2
        """
        super(EvaluationModel, self).__init__(
            'GaussianNoise',
            'gaussianNoiseModel',
            (FreeParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'sigma', False, 1, 0, 'INF',
                           parameter_transform=CosSqrClampTransform()),), ())

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            MOT_FLOAT_TYPE ''' + fname + '''(const optimize_data* const data, MOT_FLOAT_TYPE* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) - ''' + eval_fname + '''(data, x, i), 2);
                }
                return sum / (2 * pown(GaussianNoise_sigma, 2));
            }
        '''

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            double ''' + fname + '''(const optimize_data* const data, const MOT_FLOAT_TYPE* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) - ''' + eval_fname + '''(data, x, i), 2);
                }

                return - sum / (2 * pown(GaussianNoise_sigma, 2));
            }
        '''

    def set_noise_level_std(self, noise_std):
        self.parameter_list[0].value = noise_std


class OffsetGaussianEvaluationModel(EvaluationModel):

    def __init__(self):
        """Evaluates the distance between the estimated signal and the data using the Offset Gaussian evaluation.

        The offset gaussian noise is implemented as:

        sum((observation - sqrt(evaluation^2 + sigma^2))^2) / sigma^2
        """
        super(EvaluationModel, self).__init__(
            'OffsetGaussianNoise',
            'offsetGaussianNoiseModel',
            (FreeParameter(DataType.from_string('MOT_FLOAT_TYPE'), 'sigma', False, 1, 0, 'INF',
                           parameter_transform=CosSqrClampTransform()),), ())

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            MOT_FLOAT_TYPE ''' + fname + '''(const optimize_data* const data, MOT_FLOAT_TYPE* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) -
                                    sqrt(pown(''' + eval_fname + '''(data, x, i), 2) +
                                                pown(OffsetGaussianNoise_sigma, 2)), 2);
                }
                return (MOT_FLOAT_TYPE) (sum / (pown(OffsetGaussianNoise_sigma, 2)));
            }
        '''

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            double ''' + fname + '''(const optimize_data* const data, const MOT_FLOAT_TYPE* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) -
                                    sqrt(pown(''' + eval_fname + '''(data, x, i), 2) +
                                                pown(OffsetGaussianNoise_sigma, 2)), 2);
                }
                return - sum / (pown(OffsetGaussianNoise_sigma, 2));
            }
        '''

    def set_noise_level_std(self, noise_std):
        self.parameter_list[0].value = noise_std
