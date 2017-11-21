import numpy as np
from mot.cl_data_type import SimpleCLDataType
from mot.cl_function import CLFunction, SimpleCLFunction, CLPrototype, SimpleCLPrototype
from mot.model_building.parameter_functions.numdiff_info import SimpleNumDiffInfo
from mot.model_building.parameters import FreeParameter
from mot.model_building.parameter_functions.priors import ARDGaussian, UniformWithinBoundsPrior, ARDBeta
from mot.model_building.parameter_functions.proposals import GaussianProposal
from mot.model_building.parameter_functions.transformations import ClampTransform, CosSqrClampTransform

__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ModelCLPrototype(CLPrototype):

    @property
    def name(self):
        """Get the name of this model function.

        Returns:
            str: The name of this model function.
        """
        raise NotImplementedError()

    def get_free_parameters(self):
        """Get all the free parameters in this model

        Returns:
            list of CLFunctionParameter: list of all the model parameters of type FreeParameter in this model
        """
        raise NotImplementedError()


class SampleModelCLPrototype(ModelCLPrototype):

    def get_prior_parameters(self, parameter):
        """Get the prior parameters of the given parameter.

        Args:
            parameter (FreeParameter): one of the parameters of this model function

        Returns:
            list of parameters: the list of prior parameters for the given parameter
        """
        raise NotImplementedError()

    def get_model_function_priors(self):
        """Get all the model function priors.

        Returns:
            list[mot.cl_function.CLFunction]: the priors for this model function,
                these are supposed to be used in conjunction to the parameter priors.
        """
        raise NotImplementedError()


class ModelCLFunction(CLFunction, ModelCLPrototype):
    """Interface for a basic model function just for optimization purposes.

    If you need to sample the model, use the extended version of this interface :class:`SampleModelCLFunction`.
    """


class SampleModelCLFunction(ModelCLFunction, SampleModelCLPrototype):
    """Extended version of a model function for use in sampling.

    This adds functions to retrieve priors about this function.
    """


class SimpleSampleModelCLPrototype(SimpleCLPrototype, SampleModelCLPrototype):

    def __init__(self, return_type, name, cl_function_name, parameter_list, model_function_priors=None):
        """This CL function is for all estimable models

        Args:
            return_type (str): the CL return type of the function
            name (str): The name of the model
            cl_function_name (string): The name of the CL function
            parameter_list (list or tuple of CLFunctionParameter): The list of parameters required for this function
            model_function_priors (list of mot.cl_function.CLFunction): list of priors concerning this whole model
                function. The parameter names of the given functions must match those of this function.
        """
        super(SimpleSampleModelCLPrototype, self).__init__(return_type, cl_function_name, parameter_list)
        self._name = name
        self._model_function_priors = model_function_priors or []
        if isinstance(self._model_function_priors, CLFunction):
            self._model_function_priors = [self._model_function_priors]

    @property
    def name(self):
        """Get the name of this model function.

        Returns:
            str: The name of this model function.
        """
        return self._name

    def get_model_function_priors(self):
        """Get all the model function priors.

        Returns:
            list[mot.cl_function.CLFunction]: the priors for this model function, these are supposed to be used in
                conjunction to the parameter priors.
        """
        return self._model_function_priors

    def get_free_parameters(self):
        """Get all the free parameters in this model

        Returns:
            list: the list of free parameters in this model
        """
        return list([p for p in self.get_parameters() if isinstance(p, FreeParameter)])

    def get_prior_parameters(self, parameter):
        """Get the parameters referred to by the priors of the free parameters.

        This returns a list of all the parameters referenced by the prior parameters, recursively.

        Returns:
            list of parameters: the list of additional parameters in the prior for the given parameter
        """
        def get_prior_parameters(params):
            return_params = []

            for param in params:
                prior_params = param.sampling_prior.get_parameters()[3:]
                proxy_prior_params = [prior_param.get_renamed('{}.prior.{}'.format(param.name, prior_param.name))
                                      for prior_param in prior_params]

                return_params.extend(proxy_prior_params)

                free_prior_params = [p for p in proxy_prior_params if isinstance(p, FreeParameter)]
                return_params.extend(get_prior_parameters(free_prior_params))

            return return_params

        return get_prior_parameters([parameter])


class SimpleModelCLFunction(SampleModelCLFunction, SimpleCLFunction):

    def __init__(self, return_type, name, cl_function_name, parameter_list, cl_body, dependency_list=(),
                 model_function_priors=None, cl_extra=None):
        """This CL function is for all estimable models

        Args:
            return_type (str): the CL return type of the function
            name (str): The name of the model
            cl_function_name (string): The name of the CL function
            parameter_list (list or tuple of CLFunctionParameter): The list of parameters required for this function
            cl_body (str): the cl body of this function
            dependency_list (list or tuple of CLFunction): The list of CL libraries this function depends on
            model_function_priors (list of mot.cl_function.CLFunction): list of priors concerning this whole model
                function. The parameter names of the given functions must match those of this function.
        """
        super(SimpleModelCLFunction, self).__init__(return_type, cl_function_name, parameter_list,
                                                    cl_body, dependency_list=dependency_list, cl_extra=cl_extra)

        self._prototype = SimpleSampleModelCLPrototype(return_type, name, cl_function_name,
                                                       parameter_list, model_function_priors=model_function_priors)

    @property
    def name(self):
        return self._prototype.name

    def get_model_function_priors(self):
        return self._prototype.get_model_function_priors()

    def get_free_parameters(self):
        return self._prototype.get_free_parameters()

    def get_prior_parameters(self, parameter):
        return self._prototype.get_prior_parameters(parameter)


class Scalar(SimpleModelCLFunction):

    def __init__(self, name='Scalar', param_name='s', value=0.0, lower_bound=0.0, upper_bound=np.inf,
                 parameter_kwargs=None):
        """A Scalar model function to be used during optimization.

        Args:
            name (str): The name of the model
            value (number or ndarray): The initial value for the single free parameter of this function.
            lower_bound (number or ndarray): The initial lower bound for the single free parameter of this function.
            upper_bound (number or ndarray): The initial upper bound for the single free parameter of this function.
            parameter_kwargs (dict): additional settings for the parameter initialization
        """
        parameter_settings = dict(parameter_transform=ClampTransform(),
                                  sampling_proposal=GaussianProposal(1.0))
        parameter_settings.update(parameter_kwargs or {})

        super(Scalar, self).__init__(
            'mot_float_type',
            name,
            'Scalar',
            (FreeParameter(SimpleCLDataType.from_string('mot_float_type'), param_name,
                           False, value, lower_bound, upper_bound, **parameter_settings),),
            'return ' + param_name + ';')


class Weight(Scalar):

    def __init__(self, name='Weight', value=0.5, lower_bound=0.0, upper_bound=1.0, parameter_kwargs=None):
        """Implements Scalar model function to add the semantics of representing a Weight.

        Some of the code checks for type Weight, be sure to use this model function if you want to represent a Weight.

        A weight is meant to be a model volume fraction.

        Args:
            name (str): The name of the model
            value (number or ndarray): The initial value for the single free parameter of this function.
            lower_bound (number or ndarray): The initial lower bound for the single free parameter of this function.
            upper_bound (number or ndarray): The initial upper bound for the single free parameter of this function.
        """
        parameter_settings = dict(parameter_transform=CosSqrClampTransform(),
                                  sampling_proposal=GaussianProposal(0.01),
                                  sampling_prior=UniformWithinBoundsPrior(),
                                  numdiff_info=SimpleNumDiffInfo(scale_factor=10)
                                  )
        parameter_settings.update(parameter_kwargs or {})

        super(Weight, self).__init__(name=name, param_name='w', value=value, lower_bound=lower_bound,
                                     upper_bound=upper_bound, parameter_kwargs=parameter_settings)


class ARD_Beta_Weight(Weight):

    def __init__(self, name='ARD_Beta_Weight', value=0.5, lower_bound=0.0, upper_bound=1.0):
        """A compartment weight with a Beta prior, to be used in Automatic Relevance Detection

        It is exactly the same as a weight, except that it has a different prior, a Beta distribution prior between
        [0, 1].

        Args:
            name (str): The name of the model
            value (number or ndarray): The initial value for the single free parameter of this function.
            lower_bound (number or ndarray): The initial lower bound for the single free parameter of this function.
            upper_bound (number or ndarray): The initial upper bound for the single free parameter of this function.
        """
        parameter_settings = dict(sampling_prior=ARDBeta())

        super(ARD_Beta_Weight, self).__init__(name=name, value=value, lower_bound=lower_bound,
                                              upper_bound=upper_bound, parameter_kwargs=parameter_settings)


class ARD_Gaussian_Weight(Weight):

    def __init__(self, name='ARD_Gaussian_Weight', value=0.5, lower_bound=0.0, upper_bound=1.0):
        """A compartment weight with a Gaussian prior, to be used in Automatic Relevance Detection

        It is exactly the same as a weight, except that it has a different prior, a Gaussian prior with mean at zero
        and std given by a hyperparameter.

        Args:
            name (str): The name of the model
            value (number or ndarray): The initial value for the single free parameter of this function.
            lower_bound (number or ndarray): The initial lower bound for the single free parameter of this function.
            upper_bound (number or ndarray): The initial upper bound for the single free parameter of this function.
        """
        parameter_settings = dict(sampling_prior=ARDGaussian())

        super(ARD_Gaussian_Weight, self).__init__(name=name, value=value, lower_bound=lower_bound,
                                                  upper_bound=upper_bound, parameter_kwargs=parameter_settings)
