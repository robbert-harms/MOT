from copy import copy

from mot.cl_data_type import SimpleCLDataType
from mot.model_building.parameter_functions.priors import UniformWithinBoundsPrior
from mot.model_building.parameter_functions.proposals import GaussianProposal
from mot.model_building.parameter_functions.sample_statistics import GaussianPSS
from mot.model_building.parameter_functions.transformations import IdentityTransform

__author__ = 'Robbert Harms'
__date__ = "2016-10-03"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class CLFunctionParameter(object):

    def __init__(self, data_type, name):
        """Creates a new function parameter for the CL functions.

        Args:
            data_type (mot.cl_data_type.SimpleCLDataType): the data type expected by this parameter
            name (str): The name of this parameter

        Attributes:
            name (str): The name of this parameter
        """
        self._data_type = data_type
        self.name = name

    @property
    def data_type(self):
        """Get the CL data type of this parameter

        Returns:
            str: The CL data type.
        """
        return self._data_type

    @property
    def is_cl_vector_type(self):
        """Parse the data_type to see if this parameter holds a vector type (in CL)

        Returns:
            bool: True if the type of this function parameter is a CL vector type.

            CL vector types are recognized by an integer after the data type. For example: double4 is a
            CL vector type with 4 doubles.
        """
        return self._data_type.is_vector_type

    def get_renamed(self, name):
        """Get a copy of the current parameter but then with a new name.

        Args:
            name (str): the new name for this parameter

        Returns:
            cls: a copy of the current type but with a new name
        """
        new_param = copy(self)
        new_param.name = name
        return new_param


class CurrentObservationParam(CLFunctionParameter):

    def __init__(self, name='_observation'):
        """This parameter indicates that the model should inject the current observation value in the model.

        Sometimes during model linearization or other mathematical operations the current observation appears on
        both sides of the optimization equation. That is, it sometimes happens you want to use the current observation
        to model that same observation. This parameter is a signal to the model builder to inject the current
        observation.

        You can use this parameter by adding it to your model and then use the current name in your model equation.
        """
        super(CurrentObservationParam, self).__init__(SimpleCLDataType.from_string('mot_float_type'), name)


class StaticMapParameter(CLFunctionParameter):

    def __init__(self, data_type, name, value):
        """This parameter is meant for static data that is different per problem.

        These parameters are in usage similar to fixed free parameters. They are defined as static data parameters to
        make clear that they are meant to provide additional observational data.

        They differ from the model data parameters in that those are meant for data that define a model, irrespective
        of the data that is trying to be optimized. The static data parameters are supportive data about the problems
        and differ per problem instance. This makes them differ slightly in semantics.

        Args:
            data_type (mot.cl_data_type.SimpleCLDataType): the data type expected by this parameter
            name (str): The name of this parameter
            value (double or ndarray): A single value for all voxels or a list of values for each voxel

        Attributes:
            value (double or ndarray): A single value for all voxels or a list of values for each voxel
        """
        super(StaticMapParameter, self).__init__(data_type, name)
        self.value = value


class ProtocolParameter(CLFunctionParameter):
    """A protocol data parameter indicates that this parameter is supposed to be fixed using the Protocol data.

    This class of parameters is used for parameters that are constant per problem instance, but differ for the different
    measurement points (in diffusion MRI these are called the Protocol parameters).
    """


class ModelDataParameter(CLFunctionParameter):

    def __init__(self, data_type, name, value):
        """This parameter is meant for data that changes the way a model function behaves.

        These parameters are fixed and remain constant for every problem instance (voxels in DMRI)
        and for every measurement point (protocol in DMRI). They can consist of vector and array types.

        Args:
            data_type (mot.cl_data_type.SimpleCLDataType): the data type expected by this parameter
            name (str): The name of this parameter
            value (double or ndarray): A single value for all voxels or a list of values for each voxel

        Attributes:
            value (double or ndarray): A single value for all voxels or a list of values for each voxel
        """
        super(ModelDataParameter, self).__init__(data_type, name)
        self.value = value


class FreeParameter(CLFunctionParameter):

    def __init__(self, data_type, name, fixed, value, lower_bound, upper_bound,
                 parameter_transform=None, sampling_proposal=None,
                 sampling_prior=None, sampling_statistics=None):
        """This are the kind of parameters that are generally meant to be optimized.

        These parameters may optionally be fixed to a value or list of values for all voxels.

        Args:
            data_type (mot.cl_data_type.SimpleCLDataType): the data type expected by this parameter
            name (str): The name of this parameter
            fixed (boolean): If this parameter is fixed to the value given
            value (double or ndarray): A single value for all voxels or a list of values for each voxel
            lower_bound (double): The lower bound of this parameter
            upper_bound (double): The upper bound of this parameter
            parameter_transform (AbstractTransformation): The parameter transformation function
            sampling_proposal (ParameterProposal): The proposal function for use in model sampling
            sampling_prior (ParameterPrior): The prior function for use in model sampling
            sampling_statistics (ParameterSampleStatistics): The statistic functions used to get
                statistics out of the samples

        Attributes:
            value (number or ndarray): The value of this state
            lower_bound (number or ndarray): The lower bound
            upper_bound (number or ndarray): The upper bound
            fixed (boolean): If this free parameter is fixed to its value.
            parameter_transform (AbstractTransformation): The parameter transformation (codec information)
            sampling_proposal (ParameterProposal): The proposal function for use in model sampling
            sampling_prior (ParameterPrior): The prior function for use in model sampling
            sampling_statistics (ParameterSampleStatistics): The statistic functions used to get
                statistics out of the samples
        """
        super(FreeParameter, self).__init__(data_type, name)
        self._value = value
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._fixed = fixed

        self._parameter_transform = parameter_transform or IdentityTransform()
        self._sampling_proposal = sampling_proposal or GaussianProposal(1.0)
        self._sampling_prior = sampling_prior or UniformWithinBoundsPrior()
        self._sampling_statistics = sampling_statistics or GaussianPSS()

    @property
    def value(self):
        return self._value

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def fixed(self):
        return self._fixed

    @property
    def parameter_transform(self):
        return self._parameter_transform

    @property
    def sampling_proposal(self):
        return self._sampling_proposal

    @property
    def sampling_prior(self):
        return self._sampling_prior

    @property
    def sampling_statistics(self):
        return self._sampling_statistics


class LibraryParameter(CLFunctionParameter):
    """Parameters of this type are used inside library functions. They are not meant to be used in Model functions.
    """
