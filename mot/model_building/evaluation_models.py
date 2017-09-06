from mot.model_building.model_functions import SimpleModelCLFunction, SimpleSampleModelCLHeader
from mot.cl_parameter import CLFunctionParameter
from mot.model_building.parameters import FreeParameter
from mot.library_functions import LogBesseli0
from mot.model_building.parameter_functions.transformations import ClampTransform


__author__ = 'Robbert Harms'
__date__ = "2014-08-05"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class EvaluationModel(object):
    """The evaluation model is the model under which you evaluate your model estimates against observations.

    This class is a proxy for the optimization evaluation function and for the log likelihood objective function.

    This normally embed the noise model assumptions of your data.
    """

    @property
    def name(self):
        """Get the name of this model. Should be the same as the name in the CL header."""
        raise NotImplementedError()

    def get_noise_std_param_name(self):
        """Get the name of the parameter that is associated with the noise standard deviation in the problem data.

        Returns:
            str: the name of the parameter that is associated with the noise_std in the problem data.
        """
        raise NotImplementedError()

    def get_minimum_likelihood_function(self):
        """Get the function to evaluate the objective for a given observation and estimate under this noise model.

        This should return the observations such that when linearly summed we get the complete objective function value.

        Returns:
            mot.model_building.model_functions.SampleModelCLFunction: The objective function for the given observation
                index under this noise model.
        """
        raise NotImplementedError()

    def get_log_likelihood_function(self, full_likelihood=True):
        """Get the function to evaluate the log likelihood for the given observations and model estimates.

        This should return the log likelihoods as such that when linearly summed they would yield the complete
        log likelihood for the model.

        Args:
            full_likelihood (boolean): if we want the function for the complete likelihood or for the simplified
                likelihood that is faster to evaluate but might be non-normalized.

        Returns:
            mot.model_building.model_functions.SampleModelCLFunction: The objective function for the given observation
                index under this noise model.
        """
        raise NotImplementedError()

    def get_function_header(self):
        """Get the cl function header for the evaluation model.

        It is assumed and required that both the minimum likelihood function as the log likelihood function
        have the same signature.

        Returns:
            mot.model_building.model_functions.SampleModelCLHeader: the CL header for the CL functions in the
                the evaluation model.
        """
        raise NotImplementedError()


class SimpleAbstractEvaluationModel(EvaluationModel):

    def __init__(self, name, cl_function_name, parameter_list, noise_std_param_name=None,
                 dependency_list=()):
        """The evaluation model is the model under which you evaluate your model estimates against observations.

        This class is a proxy for the optimization evaluation function and for the log likelihood objective function.

        This normally embed the noise model assumptions of your data.

        Args:
            name (str): the name of this evaluation model
            cl_function_name (str): the name of the function, this is not used atm
            parameter_list (list or tuple): the list of parameters this model requires to function correctly
            noise_std_param_name (str): the name of the noise sigma parameter
            dependency_list (list or tuple): the list of function dependencies
        """
        self._name = name
        self._cl_function_name = cl_function_name
        self._parameter_list = parameter_list
        self._noise_std_param_name = noise_std_param_name
        self._dependency_list = dependency_list

    @property
    def name(self):
        return self._name

    def get_noise_std_param_name(self):
        return self._noise_std_param_name

    def get_minimum_likelihood_function(self):
        return SimpleModelCLFunction('double', self.name, self._cl_function_name, self._parameter_list,
                                     self._get_minimum_likelihood_code(),
                                     dependency_list=self._dependency_list)

    def get_log_likelihood_function(self, full_likelihood=True):
        return SimpleModelCLFunction('double', self.name, self._cl_function_name, self._parameter_list,
                                     self._get_log_likelihood_code(full_likelihood),
                                     dependency_list=self._dependency_list)

    def get_function_header(self):
        return SimpleSampleModelCLHeader('double', self.name, self._cl_function_name, self._parameter_list)

    def _get_minimum_likelihood_code(self):
        raise NotImplementedError()

    def _get_log_likelihood_code(self, full_likelihood):
        raise NotImplementedError()


class SumOfSquaresEvaluationModel(SimpleAbstractEvaluationModel):

    def __init__(self):
        """Evaluates the distance between the estimated signal and the data using the sum of squared differences.

        This is implemented as:

        .. code-block:: c

            sum((observation - evaluation)^2)

        Since the optimization routines will sum the results, we only need to return:

        .. code-block:: c

            (observation - evaluation)^2

        And for sampling we return:

        .. code-block:: c

            -(observation - model_evaluation)^2
        """
        super(SumOfSquaresEvaluationModel, self).__init__(
            'SumOfSquares', 'sumOfSquares',
            [CLFunctionParameter('double', 'observation'),
             CLFunctionParameter('double', 'model_evaluation')])

    def _get_minimum_likelihood_code(self):
        return '''
            double sumOfSquares(double observation, double model_evaluation){
                return pown(observation - model_evaluation, 2);
            }
        '''

    def _get_log_likelihood_code(self, full_likelihood):
        return '''
            double sumOfSquares(double observation, double model_evaluation){
                return -pown(observation - model_evaluation, 2);
            }
        '''


class GaussianEvaluationModel(SimpleAbstractEvaluationModel):

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

        We use this directly for sampling. For the optimization routines we need to adapt since the optimizers
        are minimization routines, we need use the negative, and since the optimizers already sum the results, we
        return here:

        .. code-block:: c

            = - log(PDF)
        """
        super(GaussianEvaluationModel, self).__init__(
            'GaussianNoiseModel', 'gaussianNoise',
            [CLFunctionParameter('double', 'observation'),
             CLFunctionParameter('double', 'model_evaluation'),
             FreeParameter('mot_float_type', 'sigma', True, 1, 0, 'INFINITY', parameter_transform=ClampTransform())],
            noise_std_param_name='sigma')

    def _get_minimum_likelihood_code(self):
        """Get the Gaussian objective function.

        This omits the constant terms for speed reasons. Omitted terms are:

         .. code-block:: c

            + log(sigma * sqrt(2 * M_PI))
        """
        return '''
            double gaussianNoise(double observation, double model_evaluation, mot_float_type sigma){
                return pown(observation - model_evaluation, 2) / (2 * sigma * sigma);
            }
        '''

    def _get_log_likelihood_code(self, full_likelihood):
        return '''
            double gaussianNoise(double observation, double model_evaluation, mot_float_type sigma){
                return - pown(observation - model_evaluation, 2) / (2 * sigma * sigma) 
                    ''' + ('- log(sigma * sqrt(2 * M_PI))' if full_likelihood else '') + ''';
            }
        '''


class OffsetGaussianEvaluationModel(SimpleAbstractEvaluationModel):

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

        Which is used directly for sampling. For the maximum likelihood estimator we need to use the negative of
        this since the optimization routines are minimization routines, and since the optimization routines
        sum the results, we return here as MLE objective function:

        .. code-block:: c

            -log(PDF)
        """
        super(OffsetGaussianEvaluationModel, self).__init__(
            'OffsetGaussianNoise', 'offsetGaussian',
            [CLFunctionParameter('double', 'observation'),
             CLFunctionParameter('double', 'model_evaluation'),
             FreeParameter('mot_float_type', 'sigma', True, 1, 0, 'INFINITY', parameter_transform=ClampTransform())],
            noise_std_param_name='sigma')

    def _get_minimum_likelihood_code(self):
        """Get the Offset Gaussian objective function.

        This omits the constant terms for speed reasons. Omitted terms are:

         .. code-block:: c

            (+ log(sigma * sqrt(2 * M_PI)))
        """
        return '''
            double offsetGaussian(double observation, double model_evaluation, mot_float_type sigma){
                return pown(observation - sqrt(pown(model_evaluation, 2) + (sigma * sigma)), 2) / (2 * sigma * sigma);
            }
        '''

    def _get_log_likelihood_code(self, full_likelihood):
        return '''
            double offsetGaussian(double observation, double model_evaluation, mot_float_type sigma){
                double estimate = sqrt(pown(model_evaluation, 2) + (sigma * sigma));
                return - pown(observation - estimate, 2) / (2 * (sigma * sigma))
                    ''' + ('- log(sigma * sqrt(2 * M_PI))' if full_likelihood else '') + ''';
            }
        '''


class RicianEvaluationModel(SimpleAbstractEvaluationModel):

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

        For sampling we use this directly. For the maximum likelihood estimator we need to use the negative of
        this since the optimization routines are minimization routines, and since the optimization routines
        sum the results, we return here as MLE objective function:

        .. code-block:: c

            -log(PDF).
        """
        super(RicianEvaluationModel, self).__init__(
            'RicianNoise', 'ricianNoise',
            [CLFunctionParameter('double', 'observation'),
             CLFunctionParameter('double', 'model_evaluation'),
             FreeParameter('mot_float_type', 'sigma', True, 1, 0, 'INFINITY', parameter_transform=ClampTransform())],
            noise_std_param_name='sigma',
            dependency_list=(LogBesseli0(),))

    def _get_minimum_likelihood_code(self):
        """Get the Rician objective function.

        This omits the constant terms for speed reasons. Omitted terms are:

         .. code-block:: c

            + log(observation/sigma^2)
            - ((observation * observation) / (2 * (sigma * sigma)))

        """
        return '''
            double ricianNoise(double observation, double model_evaluation, mot_float_type sigma){
                return ((model_evaluation * model_evaluation) / (2 * sigma * sigma)) 
                        - log_bessel_i0((observation * model_evaluation) / (sigma * sigma));
            }
        '''

    def _get_log_likelihood_code(self, full_likelihood=True):
        return '''
            double ricianNoise(double observation, double model_evaluation, mot_float_type sigma){
                return log(observation / (sigma * sigma))
                        - ((observation * observation) / (2 * sigma * sigma))
                        - ((model_evaluation * model_evaluation) / (2 * sigma * sigma))
                        + log_bessel_i0((observation * model_evaluation) / (sigma * sigma));
            }
        '''
