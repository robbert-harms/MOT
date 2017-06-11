from mot.model_building.parameters import FreeParameter
from mot.model_building.model_functions import SimpleModelFunction
from mot.library_functions import Bessel
from mot.model_building.parameter_functions.transformations import ClampTransform
from mot.cl_data_type import SimpleCLDataType


__author__ = 'Robbert Harms'
__date__ = "2014-08-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class EvaluationModel(SimpleModelFunction):

    def __init__(self, name, cl_function_name, parameter_list, dependency_list=()):
        """The evaluation model is the model under which you evaluate the estimated results against the data.

        This normally embed the noise model assumptions of your data.

        Args:
            name (str): the name of this evaluation model
            cl_function_name (str): the name of the function, this is not used atm
            parameter_list (list or tuple): the list of parameters this model requires to function correctly
            dependency_list (list or tuple): some dependencies for this model
        """
        super(EvaluationModel, self).__init__('double', name, cl_function_name, parameter_list,
                                              dependency_list=dependency_list)

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        """Get the cl code for the objective function under the given noise model.

        Args:
            fname (str): the name of the resulting function
            inst_per_problem (int): the number of instances per problem
            eval_fname (str): the name of the function that can be called to get the evaluation, its signature is:

                .. code-block:: c

                    double <fname>(const void* data, const mot_float_type* x, const uint observation_index);

            obs_fname (str): the name of the function that can be called for the observed data, its signature is:

                .. code-block:: c

                    double <fname>(const void* data, const uint observation_index);

            param_listing (str): the parameter listings for the parameters of the noise model

        Returns:
            str: The objective function under this noise model, its signature is:

                .. code-block:: c

                    double <fname>(const void* const data, mot_float_type* const x);

            That is, it always returns a double since the summations may get large.
        """
        raise NotImplementedError()

    def get_objective_per_observation_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        """Get the cl code for the objective function for a given instance under the given noise model.

        This function is used by some evaluation routines (like for example LevenbergMarquardt) that need
        a list of objective values (one per instance point), instead of a single objective function scalar.
        This function provides the information to build that list.

        Args:
            fname (str): the name of the resulting function
            inst_per_problem (int): the number of instances per problem
            eval_fname (str): the name of the function that can be called to get the evaluation, its signature is:

                .. code-block:: c

                    double <fname>(const void* data, const mot_float_type* x, const uint observation_index);

            obs_fname (str): the name of the function that can be called for the observed data, its signature is:

                .. code-block:: c

                    double <fname>(const void* data, const uint observation_index);

            param_listing (str): the parameter listings for the parameters of the noise model

        Returns:
            str: The objective function for the given observation index under this noise model, its signature is:

                .. code-block:: c

                    double (const void* const data, mot_float_type* const x, const uint observation_index);
        """
        raise NotImplementedError()

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing,
                                    full_likelihood=True):
        """Get the cl code for the log likelihood function under the given noise model.

        Args:
            fname (str): the name of the resulting function
            inst_per_problem (int): the number of instances per problem
            eval_fname (str): the name of the function that can be called to get the evaluation, its signature is:

                .. code-block:: c

                    double <fname>(const void* data, const mot_float_type* x, const uint observation_index);

            obs_fname (str): the name of the function that can be called for the observed data, its signature is:

                .. code-block:: c

                    double <fname>(const void* data, const uint observation_index);

            param_listing (str): the parameter listings for the parameters of the noise model
            full_likelihood (boolean): if we want the complete likelihood, or if we can drop the constant terms.
                The default is the complete likelihood. Disable for speed.

        Returns:
            str: the objective function under this noise model, its signature is:

                .. code-block:: c

                    double <fname>(const void* const data, mot_float_type* const x);

            That is, it always returns a double since the summations may get large.
        """
        raise NotImplementedError()

    def get_log_likelihood_per_observation_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing,
                                                    full_likelihood=True):
        """Get the cl code for the log likelihood function under the given noise model for the given observation index.

        Args:
            fname (str): the name of the resulting function
            inst_per_problem (int): the number of instances per problem
            eval_fname (str): the name of the function that can be called to get the evaluation, its signature is:

                .. code-block:: c

                    double <fname>(const void* data, const mot_float_type* x, const uint observation_index);

            obs_fname (str): the name of the function that can be called for the observed data, its signature is:

                .. code-block:: c

                    double <fname>(const void* data, const uint observation_index);

            param_listing (str): the parameter listings for the parameters of the noise model
            full_likelihood (boolean): if we want the complete likelihood, or if we can drop the constant terms.
                The default is the complete likelihood. Disable for speed.

        Returns:
            str: the objective function under this noise model, its signature is:

                .. code-block:: c

                    double <fname>(const void* const data, mot_float_type* const x,
                                   const uint observation_index);
        """
        raise NotImplementedError()

    def get_noise_std_param_name(self):
        """Get the name of the parameter that is associated with the noise standard deviation in the problem data.

        Returns:
            str: the name of the parameter that is associated with the noise_std in the problem data.
        """
        return 'sigma'


class SumOfSquaresEvaluationModel(EvaluationModel):

    def __init__(self):
        """Evaluates the distance between the estimated signal and the data using the sum of squared differences.

        This is implemented as::

            sum((observation - evaluation)^2)
        """
        super(EvaluationModel, self).__init__('SumOfSquaresNoise', 'sumOfSquaresNoise', (), ())

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            double ''' + fname + '''(const void* const data, mot_float_type* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(uint i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) - ''' + eval_fname + '''(data, x, i), 2);
                }
                return sum;
            }
        '''

    def get_objective_per_observation_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            double ''' + fname + '''(const void* const data, mot_float_type* const x,
                                     const uint observation_index){
                ''' + param_listing + '''
                return ''' + obs_fname + '''(data, observation_index) -
                        ''' + eval_fname + '''(data, x, observation_index);
            }
        '''

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing,
                                    full_likelihood=True):
        return '''
            double ''' + fname + '''(const void* const data, const mot_float_type* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(uint i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) - ''' + eval_fname + '''(data, x, i), 2);
                }
                return - sum;
            }
        '''

    def get_log_likelihood_per_observation_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing,
                                                    full_likelihood=True):
        return '''
            double ''' + fname + '''(const void* const data, const mot_float_type* const x,
                                     const uint observation_index){
                ''' + param_listing + '''
                return - (pown(''' + obs_fname + '''(data, observation_index)
                            - ''' + eval_fname + '''(data, x, observation_index), 2));
            }
        '''


class GaussianEvaluationModel(EvaluationModel):

    def __init__(self):
        """This uses the log of the Gaussian PDF for the maximum likelihood estimator and for the log likelihood.

        The PDF is defined as:

        .. code-block:: c

            PDF = 1/(sigma * sqrt(2*pi)) * exp(-(observation - evaluation)^2 / (2 * sigma^2))

        To have the joined probability over all instances one would normally have to take the product
        over all ``n`` instances:

        .. code-block:: c

            product(PDF)

        Instead of taking the product of this PDF we take the sum of the log of the PDF:

        .. code-block:: c

            sum(log(PDF))

        Where the log of the PDF is given by:

        .. code-block:: c

            log(PDF) = - ((observation - evaluation)^2 / (2 * sigma^2)) - log(sigma * sqrt(2*pi))


        For the maximum likelihood estimator we then need to use the negative of this sum:

        .. code-block:: c

            - sum(log(PDF)).
        """
        super(GaussianEvaluationModel, self).__init__(
            'GaussianNoise',
            'gaussianNoiseModel',
            (FreeParameter(SimpleCLDataType.from_string('mot_float_type'), 'sigma', True, 1, 0, 'INFINITY',
                           parameter_transform=ClampTransform()),), ())

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        """Get the Gaussian objective function.

        This omits the constant terms for speed reasons. Omitted terms are:

         .. code-block:: c

            + log(GaussianNoise_sigma * sqrt(2 * M_PI))

        """
        return '''
            double ''' + fname + '''(const void* const data, mot_float_type* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(uint i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) - ''' + eval_fname + '''(data, x, i), 2);
                }
                return sum / (2 * GaussianNoise_sigma * GaussianNoise_sigma);
            }
        '''

    def get_objective_per_observation_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            double ''' + fname + '''(const void* const data, mot_float_type* const x,
                                     const uint observation_index){
                ''' + param_listing + '''
                return ''' + obs_fname + '''(data, observation_index) -
                        ''' + eval_fname + '''(data, x, observation_index);
            }
        '''

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing,
                                    full_likelihood=True):
        return '''
            double ''' + fname + '''(const void* const data, const mot_float_type* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(uint i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) - ''' + eval_fname + '''(data, x, i), 2);
                }
                return - sum / (2 * GaussianNoise_sigma * GaussianNoise_sigma)
                    ''' + ('-' + str(inst_per_problem) + ' * log(GaussianNoise_sigma * sqrt(2 * M_PI))'
                           if full_likelihood else '') + ''';

            }
        '''

    def get_log_likelihood_per_observation_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing,
                                                    full_likelihood=True):
        return '''
            double ''' + fname + '''(const void* const data, const mot_float_type* const x,
                                     const uint observation_index){
                ''' + param_listing + '''
                return - pown(''' + obs_fname + '''(data, observation_index)
                                - ''' + eval_fname + '''(data, x, observation_index), 2)
                    / (2 * GaussianNoise_sigma * GaussianNoise_sigma)
                    ''' + ('-' + str(inst_per_problem) + ' * log(GaussianNoise_sigma * sqrt(2 * M_PI))'
                           if full_likelihood else '') + ''';
            }
        '''


class OffsetGaussianEvaluationModel(EvaluationModel):

    def __init__(self):
        """This uses the log of the Gaussian PDF for the maximum likelihood estimator and for the log likelihood.

        The PDF is defined as:

        .. code-block:: c

            PDF = 1/(sigma * sqrt(2*pi)) * exp(-(observation - sqrt(evaluation^2 + sigma^2))^2 / (2 * sigma^2))

        To have the joined probability over all instances one would have to take the product over all n instances:

        .. code-block:: c

            product(PDF)

        Instead of taking the product of this PDF we take the sum of the log of the PDF:

        .. code-block:: c

            sum(log(PDF))

        Where the log of the PDF is given by:

        .. code-block:: c

            log(PDF) = - ((observation - sqrt(evaluation^2 + sigma^2))^2 / (2 * sigma^2)) - log(sigma * sqrt(2*pi))

        For the maximum likelihood estimator we use the negative of this sum:

        .. code-block:: c

            -sum_n(log(PDF)).
        """
        super(OffsetGaussianEvaluationModel, self).__init__(
            'OffsetGaussianNoise',
            'offsetGaussianNoiseModel',
            (FreeParameter(SimpleCLDataType.from_string('mot_float_type'), 'sigma', True, 1, 0, 'INFINITY',
                           parameter_transform=ClampTransform()),), ())

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        """Get the Offset Gaussian objective function.

        This omits the constant terms for speed reasons. Omitted terms are:

         .. code-block:: c

            (+ log(OffsetGaussianNoise_sigma * sqrt(2 * M_PI)))
        """
        return '''
            double ''' + fname + '''(const void* const data, mot_float_type* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(uint i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) -
                                sqrt(pown(''' + eval_fname + '''(data, x, i), 2) +
                                    (OffsetGaussianNoise_sigma * OffsetGaussianNoise_sigma)), 2);
                }
                return sum / (2 * pown(OffsetGaussianNoise_sigma, 2));
            }
        '''

    def get_objective_per_observation_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            double ''' + fname + '''(const void* const data, mot_float_type* const x,
                                     const uint observation_index){
                ''' + param_listing + '''
                return ''' + obs_fname + '''(data, observation_index) -
                         sqrt(pown(''' + eval_fname + '''(data, x, observation_index), 2)
                                + (OffsetGaussianNoise_sigma * OffsetGaussianNoise_sigma));
            }
        '''

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing,
                                    full_likelihood=True):
        return '''
            double ''' + fname + '''(const void* const data, mot_float_type* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                for(uint i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    sum += pown(''' + obs_fname + '''(data, i) -
                                sqrt(pown(''' + eval_fname + '''(data, x, i), 2) +
                                    (OffsetGaussianNoise_sigma * OffsetGaussianNoise_sigma)), 2);
                }
                return - sum / (2 * pown(OffsetGaussianNoise_sigma, 2))
                    ''' + ('-' + str(inst_per_problem) + ' * log(OffsetGaussianNoise_sigma * sqrt(2 * M_PI))'
                           if full_likelihood else '') + ''';
            }
        '''

    def get_log_likelihood_per_observation_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing,
                                                    full_likelihood=True):
        return '''
            double ''' + fname + '''(const void* const data, mot_float_type* const x, const uint observation_index){
                ''' + param_listing + '''
                return - (pown(''' + obs_fname + '''(data, observation_index) -
                                sqrt(pown(''' + eval_fname + '''(data, x, observation_index), 2) +
                                    (OffsetGaussianNoise_sigma * OffsetGaussianNoise_sigma)), 2)) /
                            (2 * pown(OffsetGaussianNoise_sigma, 2))
                    ''' + ('-' + str(inst_per_problem) + ' * log(OffsetGaussianNoise_sigma * sqrt(2 * M_PI))'
                           if full_likelihood else '') + ''';
            }
        '''


class RicianEvaluationModel(EvaluationModel):

    def __init__(self):
        """This uses the log of the Rice PDF for the maximum likelihood estimator and for the log likelihood.

        The PDF is defined as:

        .. code-block:: c

            PDF = (observation/sigma^2)
                    * exp(-(observation^2 + evaluation^2) / (2 * sigma^2))
                    * bessel_i0((observation * evaluation) / sigma^2)

        Where where ``bessel_i0(z)`` is the modified Bessel function of the first kind with order zero. To have the
        joined probability over all instances one would have to take the product over all n instances:

        .. code-block:: c

            product(PDF)

        Instead of taking the product of this PDF over all instances we take the sum of the log of the PDF:

        .. code-block:: c

            sum(log(PDF))

        Where the log of the PDF is given by:

        .. code-block:: c

            log(PDF) = log(observation/sigma^2)
                        - (observation^2 + evaluation^2) / (2 * sigma^2)
                        + log(bessel_i0((observation * evaluation) / sigma^2))

        For the maximum likelihood estimator we use the negative of this sum:

        .. code-block:: c

            -sum(log(PDF)).
        """
        super(RicianEvaluationModel, self).__init__(
            'RicianNoise',
            'ricianNoiseModel',
            (FreeParameter(SimpleCLDataType.from_string('mot_float_type'), 'sigma', True, 1, 0, 'INFINITY',
                           parameter_transform=ClampTransform()),),
            (Bessel(),))

    def get_objective_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        """Get the Rician objective function.

        This omits the constant terms for speed reasons. Omitted terms are:

         .. code-block:: c

            + log(observation / (RicianNoise_sigma * RicianNoise_sigma))
            - ((observation * observation) / (2 * (RicianNoise_sigma * RicianNoise_sigma)))

        """
        return '''
            double ''' + fname + '''(const void* const data, mot_float_type* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                double observation;
                double evaluation;
                for(uint i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    observation = (double)''' + obs_fname + '''(data, i);
                    evaluation = (double)''' + eval_fname + '''(data, x, i);

                    sum +=  - ((evaluation * evaluation) / (2 * RicianNoise_sigma * RicianNoise_sigma))
                            + log_bessel_i0((observation * evaluation) / (RicianNoise_sigma * RicianNoise_sigma));
                }
                return -sum;
            }
        '''

    def get_objective_per_observation_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing):
        return '''
            double ''' + fname + '''(const void* const data, mot_float_type* const x,
                                     const uint observation_index){
                ''' + param_listing + '''

                double observation = (double)''' + obs_fname + '''(data, observation_index);
                double evaluation = (double)''' + eval_fname + '''(data, x, observation_index);

                return - ((evaluation * evaluation) / (2 * RicianNoise_sigma * RicianNoise_sigma))
                            + log_bessel_i0((observation * evaluation) / (RicianNoise_sigma * RicianNoise_sigma));
            }
        '''

    def get_log_likelihood_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing,
                                    full_likelihood=True):
        return '''
            double ''' + fname + '''(const void* const data, const mot_float_type* const x){
                ''' + param_listing + '''
                double sum = 0.0;
                double observation;
                double evaluation;
                for(uint i = 0; i < ''' + str(inst_per_problem) + '''; i++){
                    observation = (double)''' + obs_fname + '''(data, i);
                    evaluation = (double)''' + eval_fname + '''(data, x, i);

                    sum += log(observation / (RicianNoise_sigma * RicianNoise_sigma))
                            - ((observation * observation) / (2 * RicianNoise_sigma * RicianNoise_sigma))
                            - ((evaluation * evaluation) / (2 * RicianNoise_sigma * RicianNoise_sigma))
                            + log_bessel_i0((observation * evaluation) / (RicianNoise_sigma * RicianNoise_sigma));
                }
                return sum;
            }
        '''

    def get_log_likelihood_per_observation_function(self, fname, inst_per_problem, eval_fname, obs_fname, param_listing,
                                                    full_likelihood=True):
        return '''
            double ''' + fname + '''(const void* const data, const mot_float_type* const x,
                                     const uint observation_index){
                ''' + param_listing + '''
                double observation = (double)''' + obs_fname + '''(data, observation_index);
                double evaluation = (double)''' + eval_fname + '''(data, x, observation_index);

                return log(observation / (RicianNoise_sigma * RicianNoise_sigma))
                        - ((observation * observation) / (2 * RicianNoise_sigma * RicianNoise_sigma))
                        - ((evaluation * evaluation) / (2 * RicianNoise_sigma * RicianNoise_sigma))
                        + log_bessel_i0((observation * evaluation) / (RicianNoise_sigma * RicianNoise_sigma));
            }
        '''
