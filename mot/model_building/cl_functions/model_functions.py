import os
from pkg_resources import resource_filename
from mot.cl_data_type import CLDataType
from mot.model_building.cl_functions.base import ModelFunction
from mot.model_building.cl_functions.parameters import FreeParameter
from mot.model_building.parameter_functions.priors import ARDGaussian, UniformWithinBoundsPrior
from mot.model_building.parameter_functions.proposals import GaussianProposal
from mot.model_building.parameter_functions.transformations import ClampTransform, CosSqrClampTransform

__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Scalar(ModelFunction):

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
            name,
            'cmScalar',
            (FreeParameter(CLDataType.from_string('mot_float_type'), param_name,
                           False, value, lower_bound, upper_bound, **parameter_settings),))

    def get_cl_header(self):
        """See base class for details"""
        path = resource_filename('mot', 'data/opencl/model_functions/Scalar.h')
        return self._get_cl_dependency_headers() + "\n" + open(os.path.abspath(path), 'r').read()

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
