import os

import numpy as np
from pkg_resources import resource_filename
from mot.cl_data_type import CLDataType
from mot.model_building.cl_functions.base import ModelFunction
from mot.model_building.cl_functions.parameters import FreeParameter
from mot.model_building.parameter_functions.proposals import GaussianProposal
from mot.model_building.parameter_functions.transformations import ClampTransform, CosSqrTransform

__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Scalar(ModelFunction):

    def __init__(self, name='Scalar', value=0.0, lower_bound=0.0, upper_bound=float('inf')):
        """A Scalar model function to be used during optimization.

        Args:
            name (str): The name of the model
            value (number or ndarray): The initial value for the single free parameter of this function.
            lower_bound (number or ndarray): The initial lower bound for the single free parameter of this function.
            upper_bound (number or ndarray): The initial upper bound for the single free parameter of this function.
        """
        super(Scalar, self).__init__(
            name,
            'cmScalar',
            (FreeParameter(CLDataType.from_string('mot_float_type'), 's', False, value, lower_bound, upper_bound,
                           parameter_transform=ClampTransform(),
                           sampling_proposal=GaussianProposal(1.0)),))

    def get_cl_header(self):
        """See base class for details"""
        path = resource_filename('mot', 'data/opencl/modelFunctions/Scalar.h')
        return self._get_cl_dependency_headers() + "\n" + open(os.path.abspath(path), 'r').read()

    def get_cl_code(self):
        """See base class for details"""
        path = resource_filename('mot', 'data/opencl/modelFunctions/Scalar.cl')
        return self._get_cl_dependency_code() + "\n" + open(os.path.abspath(path), 'r').read()


class Weight(Scalar):

    def __init__(self, name='Weight', value=0.5, lower_bound=0.0, upper_bound=1.0):
        """Implements Scalar model function to add the semantics of representing a Weight.

        Some of the code checks for type Weight, be sure to use this model function if you want to represent a Weight.

        A weight is meant to be a model fraction.

        Args:
            name (str): The name of the model
            value (number or ndarray): The initial value for the single free parameter of this function.
            lower_bound (number or ndarray): The initial lower bound for the single free parameter of this function.
            upper_bound (number or ndarray): The initial upper bound for the single free parameter of this function.
        """
        super(Weight, self).__init__(name=name, value=value, lower_bound=lower_bound, upper_bound=upper_bound)
        self.parameter_list[0].name = 'w'
        self.parameter_list[0].parameter_transform = CosSqrTransform()
        self.parameter_list[0].sampling_proposal = GaussianProposal(0.01)
        self.parameter_list[0].perturbation_function = lambda v: np.clip(
            v + np.random.normal(scale=0.05, size=v.shape), 0, 1)
