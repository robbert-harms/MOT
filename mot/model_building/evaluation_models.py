from mot.model_building.model_functions import SimpleModelFunction

from mot.cl_function import SimpleCLFunction
from mot.cl_parameter import CLFunctionParameter
from mot.model_building.parameters import FreeParameter
from mot.library_functions import LogBesseli0
from mot.model_building.parameter_functions.transformations import ClampTransform


__author__ = 'Robbert Harms'
__date__ = "2014-08-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class EvaluationModel(SimpleModelFunction):

    def __init__(self, name, cl_function_name, parameter_list, noise_std_param_name=None, prior_parameters=None):
        """The evaluation model is the model under which you evaluate your model estimates against observations.

        This class is a proxy for the optimization evaluation function and for the log likelihood objective function.

        This normally embed the noise model assumptions of your data.

        Args:
            name (str): the name of this evaluation model
            cl_function_name (str): the name of the function, this is not used atm
            parameter_list (list or tuple): the list of parameters this model requires to function correctly
            prior_parameters (list or tuple): the list of prior parameters
        """
        super(EvaluationModel, self).__init__('double', name, cl_function_name, parameter_list, '')
        self._noise_std_param_name = noise_std_param_name

    def evaluate(self, inputs, double_precision=False):
        raise TypeError('The evaluation model can not be evaluated directly. '
                        'This class is only a proxy for the objective function and the log likelihood function.'
                        'Use `get_objective_per_observation_function` or `get_log_likelihood_per_observation_function`'
                        'and evaluate those.')

    def get_noise_std_param_name(self):
        """Get the name of the parameter that is associated with the noise standard deviation in the problem data.

        Returns:
            str: the name of the parameter that is associated with the noise_std in the problem data.
        """
        return self._noise_std_param_name

    def get_objective_per_observation_function(self):
        """Get the function to evaluate the objective for a given observation and estimate under this noise model.

        This should return the observations such that the square root of the squared sum would yield the
        complete objective function value. That is, the calling routines must square the output of this function.

        Returns:
            mot.cl_function.CLFunction: The objective function for the given observation index under this noise model.
        """
        raise NotImplementedError()

    def get_log_likelihood_per_observation_function(self, full_likelihood=True):
        """Get the function to evaluate the log likelihood for the given observations and model estimates.

        This should return the log likelihoods as such that when linearly summed they would yield the complete
        log likelihood for the model.

        Args:
            full_likelihood (boolean): if we want the function for the complete likelihood or for the simplified
                likelihood that is faster to evaluate but might be non-normalized.

        Returns:
            mot.cl_function.CLFunction: The objective function for the given observation index under this noise model.
        """
        raise NotImplementedError()


class SumOfSquaresEvaluationModel(EvaluationModel):

    def __init__(self):
        """Evaluates the distance between the estimated signal and the data using the sum of squared differences.

        This is implemented as:

        .. code-block:: c

            sum((observation - evaluation)^2)

        Since the optimization routines will square and sum the results, we only need to return:

        .. code-block:: c

            observation - evaluation

        And for sampling we return:

        .. code-block:: c

             -pown(observation - model_evaluation, 2)
        """
        super(SumOfSquaresEvaluationModel, self).__init__('SumOfSquaresNoise', 'sumOfSquaresNoise', (), None)

    def get_objective_per_observation_function(self):
        cl_code = '''
            double sumOfSquaresEvaluationModel(double observation, double model_evaluation){
                return observation - model_evaluation;
            }
        '''
        return SimpleCLFunction('double', 'sumOfSquaresEvaluationModel',
                                [CLFunctionParameter('double', 'observation'),
                                 CLFunctionParameter('double', 'model_evaluation')],
                                cl_code)

    def get_log_likelihood_per_observation_function(self, full_likelihood=True):
        cl_code = '''
            double sumOfSquaresLL(double observation, double model_evaluation){
                return -pown(observation - model_evaluation, 2);
            }
        '''
        return SimpleCLFunction('double', 'sumOfSquaresLL',
                                [CLFunctionParameter('double', 'observation'),
                                 CLFunctionParameter('double', 'model_evaluation')],
                                cl_code)


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


        Since the optimization routines are minimization routines, we need use the negative of this sum:

        .. code-block:: c

            = - sum(log(PDF))

        as maximum likelihood estimator for the optimization routines. Furthermore, since the optimization
        routines already square and sum the results, we return here:

        .. code-block:: c

            - sqrt(log(PDF))
        """
        super(GaussianEvaluationModel, self).__init__(
            'GaussianNoise',
            'gaussianNoiseModel',
            (FreeParameter('mot_float_type', 'sigma', True, 1, 0, 'INFINITY', parameter_transform=ClampTransform()),),
            'sigma')

    def get_objective_per_observation_function(self):
        """Get the Gaussian objective function.

        This omits the constant terms for speed reasons. Omitted terms are:

         .. code-block:: c

            + log(sigma * sqrt(2 * M_PI))
        """
        cl_code = '''
            double gaussianEvaluationModel(double observation, double model_evaluation, mot_float_type sigma){
                return (observation - model_evaluation) / (M_SQRT2 * sigma);
            }
        '''
        return SimpleCLFunction('double', 'gaussianEvaluationModel',
                                [CLFunctionParameter('double', 'observation'),
                                 CLFunctionParameter('double', 'model_evaluation'),
                                 CLFunctionParameter('mot_float_type', 'sigma')],
                                cl_code)

    def get_log_likelihood_per_observation_function(self, full_likelihood=True):
        cl_code = '''
            double gaussianEvaluationLL(double observation, double model_evaluation, mot_float_type sigma){
                return - pown(observation - model_evaluation, 2) / (2 * sigma * sigma) 
                    ''' + ('- log(sigma * sqrt(2 * M_PI))' if full_likelihood else '') + ''';
            }
        '''
        return SimpleCLFunction('double', 'gaussianEvaluationLL',
                                [CLFunctionParameter('double', 'observation'),
                                 CLFunctionParameter('double', 'model_evaluation'),
                                 CLFunctionParameter('mot_float_type', 'sigma')],
                                cl_code)


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

        For the maximum likelihood estimator we need to use the negative of the log(PDF) since the optimization routines
        are minimization routines, and since the optimization routines already take the square and sum the results,
        we return here as MLE objective function:

        .. code-block:: c

            - sqrt(log(PDF)).
        """
        super(OffsetGaussianEvaluationModel, self).__init__(
            'OffsetGaussianNoise',
            'offsetGaussianNoiseModel',
            (FreeParameter('mot_float_type', 'sigma', True, 1, 0, 'INFINITY', parameter_transform=ClampTransform()),),
            'sigma')

    def get_objective_per_observation_function(self):
        """Get the Offset Gaussian objective function.

        This omits the constant terms for speed reasons. Omitted terms are:

         .. code-block:: c

            (+ log(sigma * sqrt(2 * M_PI)))
        """
        cl_code = '''
            double offsetGaussianEvaluationModel(double observation, double model_evaluation, mot_float_type sigma){
                return (observation - sqrt(pown(model_evaluation, 2) + (sigma * sigma))) / (M_SQRT2 * sigma);
            }
        '''
        return SimpleCLFunction('double', 'offsetGaussianEvaluationModel',
                                [CLFunctionParameter('double', 'observation'),
                                 CLFunctionParameter('double', 'model_evaluation'),
                                 CLFunctionParameter('mot_float_type', 'sigma')],
                                cl_code)

    def get_log_likelihood_per_observation_function(self, full_likelihood=True):
        cl_code = '''
            double offsetGaussianEvaluationLL(double observation, double model_evaluation, mot_float_type sigma){
                double estimate = sqrt(pown(model_evaluation, 2) + (sigma * sigma));
                return - pown(observation - estimate, 2) / (2 * (sigma * sigma))
                    ''' + ('- log(sigma * sqrt(2 * M_PI))' if full_likelihood else '') + ''';
            }
        '''
        return SimpleCLFunction('double', 'offsetGaussianEvaluationLL',
                                [CLFunctionParameter('double', 'observation'),
                                 CLFunctionParameter('double', 'model_evaluation'),
                                 CLFunctionParameter('mot_float_type', 'sigma')],
                                cl_code)


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

        For the maximum likelihood estimator we need to use the negative of the log(PDF) since the optimization routines
        are minimization routines, and since the optimization routines already take the square and sum the results,
        we return here as MLE objective function:

        .. code-block:: c

            - sqrt(log(PDF)).
        """
        super(RicianEvaluationModel, self).__init__(
            'RicianNoise',
            'ricianNoiseModel',
            (FreeParameter('mot_float_type', 'sigma', True, 1, 0, 'INFINITY', parameter_transform=ClampTransform()),),
            'sigma')

    def get_objective_per_observation_function(self):
        """Get the Rician objective function.

        This omits the constant terms for speed reasons. Omitted terms are:

         .. code-block:: c

            + log(observation/sigma^2)
            - ((observation * observation) / (2 * (sigma * sigma)))

        """
        cl_code = '''
            double ricianEvaluationModel(double observation, double model_evaluation, mot_float_type sigma){
                return sqrt(log_bessel_i0((observation * model_evaluation) / (sigma * sigma)) 
                            - ((model_evaluation * model_evaluation) / (2 * sigma * sigma))
                           );
            }
        '''
        return SimpleCLFunction('double', 'ricianEvaluationModel',
                                [CLFunctionParameter('double', 'observation'),
                                 CLFunctionParameter('double', 'model_evaluation'),
                                 CLFunctionParameter('mot_float_type', 'sigma')],
                                cl_code,
                                dependency_list=(LogBesseli0(),))

    def get_log_likelihood_per_observation_function(self, full_likelihood=True):
        cl_code = '''
            double ricianEvaluationLL(double observation, double model_evaluation, mot_float_type sigma){
                return log(observation / (sigma * sigma))
                        - ((observation * observation) / (2 * sigma * sigma))
                        - ((model_evaluation * model_evaluation) / (2 * sigma * sigma))
                        + log_bessel_i0((observation * model_evaluation) / (sigma * sigma));
            }
        '''
        return SimpleCLFunction('double', 'ricianEvaluationLL',
                                [CLFunctionParameter('double', 'observation'),
                                 CLFunctionParameter('double', 'model_evaluation'),
                                 CLFunctionParameter('mot_float_type', 'sigma')],
                                cl_code,
                                dependency_list=(LogBesseli0(),))
