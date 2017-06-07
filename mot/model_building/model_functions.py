import os
from pkg_resources import resource_filename
from mot.cl_data_type import SimpleCLDataType
from mot.library_functions import CLLibrary
from mot.model_building.model_function_priors import ModelFunctionPrior
from mot.model_building.parameters import FreeParameter
from mot.model_building.parameter_functions.priors import ARDGaussian, UniformWithinBoundsPrior
from mot.model_building.parameter_functions.proposals import GaussianProposal
from mot.model_building.parameter_functions.transformations import ClampTransform, CosSqrClampTransform

__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ModelFunction(object):

    @property
    def return_type(self):
        """Get the type (in CL naming) of the returned value from this function.

        Returns:
            str: The return type of this CL function. (Examples: double, int, double4, ...)
        """
        raise NotImplementedError()

    @property
    def cl_function_name(self):
        """Return the name of the implemented CL function

        Returns:
            str: The name of this CL function
        """
        raise NotImplementedError()

    @property
    def parameter_list(self):
        """Return the list of parameters from this CL function.

        Returns:
            A list containing instances of CLFunctionParameter."""
        raise NotImplementedError()

    def get_cl_code(self):
        """Get the function code for this function and all its dependencies.

        Returns:
            str: The CL code for inclusion in a kernel.
        """
        raise NotImplementedError()

    def get_free_parameters(self):
        """Get all the free parameters in this model

        Returns:
            list: the list of free parameters in this model
        """
        raise NotImplementedError()

    def get_prior_parameters(self, parameter):
        """Get the parameters referred to by the priors of the free parameters.

        This returns a list of all the parameters referenced by the prior parameters, recursively.

        Returns:
            list of parameters: the list of additional parameters in the prior for the given parameter
        """
        raise NotImplementedError()

    def get_parameters_of_type(self, instance_types):
        """Get all parameters whose state instance is one of the given types.

        Args:
            instance_types (list of DataType class names, or single DataType classname);
                The instance type we want to get all the parameters of.

        Returns:
            A list of parameters whose type matches one or more of the given types.
        """
        raise NotImplementedError()

    def get_parameter_by_name(self, param_name):
        """Get a parameter by name.

        Args:
            param_name (str): The name of the parameter to return

        Returns:
            ClFunctionParameter: the parameter of the given name

        Raises:
            KeyError: if the parameter could not be found.
        """
        raise NotImplementedError()

    def get_cl_dependency_code(self):
        """Get the CL code for all the CL code for all the dependencies.

        Returns:
            str: The CL code with the actual code.
        """
        raise NotImplementedError()

    def get_model_function_priors(self):
        """Get all the model function priors.

        Returns:
            list of mot.model_building.model_function_priors.ModelFunctionPrior: the priors for this model function,
                these are supposed to be used in conjunction to the parameter priors.
        """
        raise NotImplementedError()


class SimpleModelFunction(ModelFunction):

    def __init__(self, return_type, name, cl_function_name, parameter_list, dependency_list=(),
                 model_function_priors=None):
        """This CL function is for all estimable models

        Args:
            return_type (str): the CL return type of the function
            name (str): The name of the model
            cl_function_name (string): The name of the CL function
            parameter_list (list or tuple of CLFunctionParameter): The list of parameters required for this function
            dependency_list (list or tuple of CLLibrary): The list of CL libraries this function depends on
            model_function_priors (list of mot.model_building.model_function_priors.ModelFunctionPrior):
                list of priors concerning this whole model function
        """
        self._name = name
        self._return_type = return_type
        self._function_name = cl_function_name
        self._parameter_list = parameter_list
        self._dependency_list = dependency_list
        self._model_function_priors = model_function_priors or []
        if isinstance(self._model_function_priors, ModelFunctionPrior):
            self._model_function_priors = [self._model_function_priors]

    @property
    def return_type(self):
        """Get the type (in CL naming) of the returned value from this function.

        Returns:
            str: The return type of this CL function. (Examples: double, int, double4, ...)
        """
        return self._return_type

    @property
    def cl_function_name(self):
        """Return the name of the implemented CL function

        Returns:
            str: The name of this CL function
        """
        return self._function_name

    @property
    def parameter_list(self):
        """Return the list of parameters from this CL function.

        Returns:
            A list containing instances of CLFunctionParameter."""
        return self._parameter_list

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
            list of mot.model_building.model_function_priors.ModelFunctionPrior: the priors for this model function,
                these are supposed to be used in conjunction to the parameter priors.
        """
        return self._model_function_priors

    def get_free_parameters(self):
        """Get all the free parameters in this model

        Returns:
            list: the list of free parameters in this model
        """
        return self.get_parameters_of_type(FreeParameter)

    def get_prior_parameters(self, parameter):
        """Get the parameters referred to by the priors of the free parameters.

        This returns a list of all the parameters referenced by the prior parameters, recursively.

        Returns:
            list of parameters: the list of additional parameters in the prior for the given parameter
        """
        def get_prior_parameters(params):
            return_params = []

            for param in params:
                prior_params = param.sampling_prior.get_parameters()
                proxy_prior_params = [prior_param.get_renamed('{}.prior.{}'.format(param.name, prior_param.name))
                                      for prior_param in prior_params]

                return_params.extend(proxy_prior_params)

                free_prior_params = [p for p in proxy_prior_params if isinstance(p, FreeParameter)]
                return_params.extend(get_prior_parameters(free_prior_params))

            return return_params

        return get_prior_parameters([parameter])

    def get_parameters_of_type(self, instance_types):
        """Get all parameters whose state instance is one of the given types.

        Args:
            instance_types (list of DataType class names, or single DataType classname);
                The instance type we want to get all the parameters of.

        Returns:
            A list of parameters whose type matches one or more of the given types.
        """
        return list([p for p in self.parameter_list if isinstance(p, instance_types)])

    def get_parameter_by_name(self, param_name):
        """Get a parameter by name.

        Args:
            param_name (str): The name of the parameter to return

        Returns:
            ClFunctionParameter: the parameter of the given name

        Raises:
            KeyError: if the parameter could not be found.
        """
        for e in self.parameter_list:
            if e.name == param_name:
                return e
        raise KeyError('The parameter with the name "{}" could not be found.'.format(param_name))

    def get_cl_dependency_code(self):
        """Get the CL code for all the CL code for all the dependencies.

        Returns:
            str: The CL code with the actual code.
        """
        return self._get_cl_dependency_code()

    def get_cl_code(self):
        """Get the function code for this function and all its dependencies.

        Returns:
            str: The CL code for inclusion in a kernel.
        """
        return ''

    def _get_cl_dependency_code(self):
        """Get the CL code for all the CL code for all the dependencies.

        Returns:
            str: The CL code with the actual code.
        """
        code = ''
        for d in self._dependency_list:
            code += d.get_cl_code() + "\n"
        return code

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return type(self) != type(other)


class Scalar(SimpleModelFunction):

    def __init__(self, name='Scalar', param_name='s', value=0.0, lower_bound=0.0, upper_bound=float('inf'),
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
            'double',
            name,
            'cmScalar',
            (FreeParameter(SimpleCLDataType.from_string('mot_float_type'), param_name,
                           False, value, lower_bound, upper_bound, **parameter_settings),))

    def get_cl_code(self):
        """See base class for details"""
        path = resource_filename('mot', 'data/opencl/model_functions/Scalar.cl')
        return self._get_cl_dependency_code() + "\n" + open(os.path.abspath(path), 'r').read()


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
                                  sampling_prior=UniformWithinBoundsPrior())
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
        parameter_settings = dict(sampling_prior=ARDGaussian())

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
