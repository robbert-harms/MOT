from mot.cl_data_type import CLDataType
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
            data_type (mot.cl_data_type.CLDataType): the data type expected by this parameter
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


class CurrentObservationParam(CLFunctionParameter):

    def __init__(self, name='_observation'):
        """This parameter indicates that the model should inject the current observation value in the model.

        Sometimes during model linearization or other mathematical operations the current observation appears on
        both sides of the optimization equation. That is, it sometimes happens you want to use the current observation
        to model that same observation. This parameter is a signal to the model builder to inject the current
        observation.

        You can use this parameter by adding it to your model and then use the current name in your model equation.
        """
        super(CurrentObservationParam, self).__init__(CLDataType.from_string('mot_float_type'), name)


class StaticMapParameter(CLFunctionParameter):

    def __init__(self, data_type, name, value):
        """This parameter is meant for static data that is different per problem.

        These parameters are in usage similar to fixed free parameters. They are defined as static data parameters to
        make clear that they are meant to provide additional observational data.

        They differ from the model data parameters in that those are meant for data that define a model, irrespective
        of the data that is trying to be optimized. The static data parameters are supportive data about the problems
        and differ per problem instance. This makes them differ slightly in semantics.

        Args:
            data_type (mot.cl_data_type.CLDataType): the data type expected by this parameter
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
            data_type (mot.cl_data_type.CLDataType): the data type expected by this parameter
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
            data_type (mot.cl_data_type.CLDataType): the data type expected by this parameter
            name (str): The name of this parameter
            fixed (boolean): Fix this parameter is fixed to the value given
            value (double or ndarray): A single value for all voxels or a list of values for each voxel
            lower_bound (double): The lower bound of this parameter
            upper_bound (double): The upper bound of this parameter
            parameter_transform (AbstractTransformation): The parameter transformation function
            sampling_proposal (AbstractParameterProposal): The proposal function for use in model sampling
            sampling_prior (AbstractParameterPrior): The prior function for use in model sampling
            sampling_statistics (ParameterSampleStatistics): The statistic functions used to get
                statistics out of the samples

        Attributes:
            value (number or ndarray): The value of this state
            lower_bound (number or ndarray): The lower bound
            upper_bound (number or ndarray): The upper bound
            fixed (boolean): If this free parameter is fixed to its value.
            parameter_transform (AbstractTransformation): The parameter transformation (codec information)
            sampling_proposal (AbstractParameterProposal): The proposal function for use in model sampling
            sampling_prior (AbstractParameterPrior): The prior function for use in model sampling
            sampling_statistics (ParameterSampleStatistics): The statistic functions used to get
                statistics out of the samples
        """
        super(FreeParameter, self).__init__(data_type, name)
        self.value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fixed = fixed

        self.parameter_transform = parameter_transform or IdentityTransform()
        self.sampling_proposal = sampling_proposal or GaussianProposal(1.0)
        self.sampling_prior = sampling_prior or UniformWithinBoundsPrior()
        self.sampling_statistics = sampling_statistics or GaussianPSS()

    def unfix(self):
        """Set the boolean fixed to false. Then return self.

        Returns:
            A reference to self for chaining.
        """
        self.fixed = False
        return self

    def fix_to(self, value):
        """Set the boolean fixed to True and set the value to the given value.

        Args:
            value (number or ndarray): The value to fix this state to.

        Returns:
            A reference to self for chaining.
        """
        self.value = value
        self.fixed = True


class LibraryParameter(CLFunctionParameter):
    """Parameters of this type are used inside library functions. They are not meant to be used in Model functions.
    """
