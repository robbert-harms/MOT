from mot.base import ModelFunction, FreeParameter
from mot.cl_functions import Bessel
from mot.model_building.parameter_functions.transformations import CosSqrClampTransform
from mot.base import CLDataType


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
                double <fname>(const optimize_data* const data, MOT_FLOAT_TYPE* const x);

            That is, it always returns a double since the summations may get large.
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

            That is, it always returns a double since the summations may get large.
        """

    def set_noise_level_std(self, noise_std, fix=False):
        """Set the estimate of the noise level standard deviation.

        We put this here as a method to make the method work with object oriented polymorphism. That is, not
        all noiselevels use the same parameter for the noise standard deviation.

        This method makes no assumptions about the state of the parameters affected, they can be fixed or not. This
        function should just set the right value.

        Args:
            noise_std (double): the noise standard deviation
            fix (boolean): if we also fix the parameter. This fixes the parameter to the given value.
                On False this does nothing and does not set fixed to False if it was already set to True.

        Returns:
            self: for chaining
        """
        return self

    def get_noise_level_std(self):
        """Get the estimate of the noise level standard deviation.

        Returns:
            noise_std (double): the noise standard deviation
        """
        return 1


class SumOfSquares(EvaluationModel):

    def __init__(self):
        """Evaluates the distance between the estimated signal and the data using the sum of squared distance.

        This is implemented as:
        sum((observation - evaluation)^2)
        """
        super(EvaluationModel, self).__init__('SumOfSquaresNoise', 'sumOfSquaresNoise', (), ())

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            double ''' + fname + '''(const optimize_data* const data, MOT_FLOAT_TYPE* const x){
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
        """This uses the log of the Gaussian PDF for the maximum likelihood estimator and for the log likelihood.

        The PDF is defined as:

            PDF = 1/(sigma * sqrt(2*pi)) * exp(-(observation - evaluation)^2 / (2 * sigma^2))

        To have the joined probability over all instances one would have to take the product over all n instances:

            prod_n(PDF)

        Instead of taking the product of this PDF we take the sum of the log of the PDF:

            sum_n(log(PDF))

        Where the log of the PDF is given by:

            log(PDF) = - ((observation - evaluation)^2 / (2 * sigma^2)) - log(sigma * sqrt(2*pi))


        For the maximum likelihood estimator we use the negative of this sum: -sum_n(log(PDF)).
        """
        super(GaussianEvaluationModel, self).__init__(
            'GaussianNoise',
            'gaussianNoiseModel',
            (FreeParameter(CLDataType.from_string('MOT_FLOAT_TYPE'), 'sigma', False, 1, 0, 'INF',
                           parameter_transform=CosSqrClampTransform()),), ())

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        # omitted constant term for speed
        # + log(GaussianNoise_sigma * sqrt(2 * M_PI));
        return '''
            double ''' + fname + '''(const optimize_data* const data, MOT_FLOAT_TYPE* const x){
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
                    sum += (pown(''' + obs_fname + '''(data, i) - ''' + eval_fname + '''(data, x, i), 2)
                                    / (2 * pown(GaussianNoise_sigma, 2)))
                             + log(GaussianNoise_sigma * sqrt(2 * M_PI));
                }
                return - sum;
            }
        '''

    def set_noise_level_std(self, noise_std, fix=False):
        self.parameter_list[0].value = noise_std

        if fix:
            self.parameter_list[0].fixed = True

        return self

    def get_noise_level_std(self):
        return self.parameter_list[0].value


class OffsetGaussianEvaluationModel(EvaluationModel):

    def __init__(self):
        """This uses the log of the Gaussian PDF for the maximum likelihood estimator and for the log likelihood.

        The PDF is defined as:

            PDF = 1/(sigma * sqrt(2*pi)) * exp(-(observation - sqrt(evaluation^2 + sigma^2))^2 / (2 * sigma^2))

        To have the joined probability over all instances one would have to take the product over all n instances:

            prod_n(PDF)

        Instead of taking the product of this PDF we take the sum of the log of the PDF:

            sum_n(log(PDF))

        Where the log of the PDF is given by:

            log(PDF) = - ((observation - sqrt(evaluation^2 + sigma^2))^2 / (2 * sigma^2)) - log(sigma * sqrt(2*pi))


        For the maximum likelihood estimator we use the negative of this sum: -sum_n(log(PDF)).
        """
        super(OffsetGaussianEvaluationModel, self).__init__(
            'OffsetGaussianNoise',
            'offsetGaussianNoiseModel',
            (FreeParameter(CLDataType.from_string('MOT_FLOAT_TYPE'), 'sigma', False, 1, 0, 'INF',
                           parameter_transform=CosSqrClampTransform()),), ())

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        # omitted constant terms for speed
        # + log(OffsetGaussianNoise_sigma * sqrt(2 * M_PI));
        return '''
            double ''' + fname + '''(const optimize_data* const data, MOT_FLOAT_TYPE* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) -
                                sqrt(pown(''' + eval_fname + '''(data, x, i), 2) +
                                    pown(OffsetGaussianNoise_sigma, 2)), 2);
                }
                return sum / (2 * pown(OffsetGaussianNoise_sigma, 2));
            }
        '''

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            double ''' + fname + '''(const optimize_data* const data, MOT_FLOAT_TYPE* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) -
                                sqrt(pown(''' + eval_fname + '''(data, x, i), 2) +
                                    pown(OffsetGaussianNoise_sigma, 2)), 2)
                                            / (2 * pown(OffsetGaussianNoise_sigma, 2))
                            + log(OffsetGaussianNoise_sigma * sqrt(2 * M_PI));
                }
                return -sum;
            }
        '''

    def set_noise_level_std(self, noise_std, fix=False):
        self.parameter_list[0].value = noise_std

        if fix:
            self.parameter_list[0].fixed = True

        return self

    def get_noise_level_std(self):
        return self.parameter_list[0].value


class RicianEvaluationModel(EvaluationModel):

    def __init__(self):
        """This uses the log of the Rice PDF for the maximum likelihood estimator and for the log likelihood.

        The PDF is defined as:

            PDF = (observation/sigma^2)
                    * exp(-(observation^2 + evaluation^2) / (2 * sigma^2))
                    * bessel_i0((observation * evaluation) / sigma^2)

        Where where bessel_i0(z) is the modified Bessel function of the first kind with order zero. To have the
        joined probability over all instances one would have to take the product over all n instances:

            prod_n(PDF)

        Instead of taking the product of this PDF over all instances we take the sum of the log of the PDF:

            sum_n(log(PDF))

        Where the log of the PDF is given by:

            log(PDF) = log(observation/sigma^2)
                        - (observation^2 + evaluation^2) / (2 * sigma^2)
                        + log(bessel_i0((observation * evaluation) / sigma^2))

        For the maximum likelihood estimator we use the negative of this sum: -sum_n(log(PDF)).
        """
        super(RicianEvaluationModel, self).__init__(
            'RicianNoise',
            'ricianNoiseModel',
            (FreeParameter(CLDataType.from_string('MOT_FLOAT_TYPE'), 'sigma', False, 1, 0, 'INF',
                           parameter_transform=CosSqrClampTransform()),),
            (Bessel(),))

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        # omitted the constant terms for speed
        # + log(observation / pown(RicianNoise_sigma, 2))
        # - (pown(observation, 2) / (2 * pown(RicianNoise_sigma, 2)))
        return '''
            double ''' + fname + '''(const optimize_data* const data, MOT_FLOAT_TYPE* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                double observation;
                double evaluation;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    observation = (double)''' + obs_fname + '''(data, i);
                    evaluation = (double)''' + eval_fname + '''(data, x, i);

                    sum +=  - (pown(evaluation, 2) / (2 * pown(RicianNoise_sigma, 2)))
                            + log_bessel_i0((observation * evaluation) / pown(RicianNoise_sigma, 2));
                }
                return -sum;
            }
        '''

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            double ''' + fname + '''(const optimize_data* const data, const MOT_FLOAT_TYPE* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                double observation;
                double evaluation;
                for(int i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    observation = (double)''' + obs_fname + '''(data, i);
                    evaluation = (double)''' + eval_fname + '''(data, x, i);

                    sum += log(observation / pown(RicianNoise_sigma, 2))
                            - (pown(observation, 2) / (2 * pown(RicianNoise_sigma, 2)))
                            - (pown(evaluation, 2) / (2 * pown(RicianNoise_sigma, 2)))
                            + log_bessel_i0((observation * evaluation) / pown(RicianNoise_sigma, 2));
                }
                return sum;
            }
        '''

    def set_noise_level_std(self, noise_std, fix=False):
        self.parameter_list[0].value = noise_std

        if fix:
            self.parameter_list[0].fixed = True

        return self

    def get_noise_level_std(self):
        return self.parameter_list[0].value
