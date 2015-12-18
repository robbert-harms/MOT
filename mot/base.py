import os
import numpy as np
from .model_building.parameter_functions.priors import UniformWithinBoundsPrior
from .model_building.parameter_functions.proposals import GaussianProposal
from .model_building.parameter_functions.sample_statistics import GaussianPSS
from .model_building.parameter_functions.transformations import IdentityTransform

__author__ = 'Robbert Harms'
__date__ = "2015-03-21"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DataType(object):

    def __init__(self, raw_data_type, is_pointer_type=False, vector_length=None,
                 address_space_qualifier=None, pre_data_type_type_qualifiers=None, post_data_type_type_qualifier=None):
        """Create a new CL data type container.

        The CL type can either be a CL native type (half, double, int, ...) or the special MOT_FLOAT_TYPE type.

        Args:
            raw_data_type (str): the specific data type without the vector number and asterisks
            numpy_type (numpy data type): the corresponding numpy data type, used in conversions
            is_pointer_type (boolean): If this parameter is a pointer type (appened by a *)
            vector_length (int or None): If None this data type is not a CL vector type.
                If it is an integer it is the vector length of this data type (2, 3, 4, ...)
            address_space_qualifier (str or None): the address space qualifier or None if not used
            pre_data_type_type_qualifiers (list of str or None): the type qualifiers to use before the data type
            post_data_type_type_qualifier (str or None): the type qualifier to use before the data type
        """
        self._raw_data_type = raw_data_type
        self._is_pointer_type = is_pointer_type
        self._vector_length = vector_length
        self.address_space_qualifier = address_space_qualifier
        self.pre_data_type_type_qualifiers = pre_data_type_type_qualifiers
        self.post_data_type_type_qualifier = post_data_type_type_qualifier

    @classmethod
    def from_string(cls, cl_type_str):
        """Generate a datatype from a CL type.

        This only accepts data types as input. If you need to add other qualifiers, please use the functions to do so.

        Args:
            cl_type_str (str): the type qualifier
        """
        address_space_qualifier = None
        pre_data_type_type_qualifier = None
        post_data_type_type_qualifier = None

        raw_type = cl_type_str.replace('*', '')
        raw_type = raw_type.replace(' ', '')
        raw_type = ''.join([i for i in raw_type if not i.isdigit()])
        vector_length = ''.join([i for i in cl_type_str if i.isdigit()])
        if vector_length == '':
            vector_length = None
        is_pointer = ('*' in cl_type_str)
        return DataType(raw_type, is_pointer_type=is_pointer, vector_length=vector_length)

    @property
    def raw_data_type(self):
        """Get the data type defined in this object

        Returns:
            str: The scalar type of this data type.
        """
        return self._raw_data_type

    @property
    def cl_type(self):
        """Get the type of this parameter in CL language

        This only returns the parameter type (like 'double' or 'int*' or 'float4*' ...). It does not include other
        qualifiers.

        Returns:
            str: The name of this data type
        """
        s = self._raw_data_type

        if self._vector_length is not None:
            s += str(self._vector_length)

        if self._is_pointer_type:
            s += '*'

        return s

    @property
    def is_vector_type(self):
        """Check if this data type is a vector type

        Returns:
            boolean: True if it is a vector type, false otherwise
        """
        return self._vector_length is not None

    @property
    def is_pointer_type(self):
        """Check if this data type is a pointer type

        Returns:
            boolean: True if it is a pointer type, false otherwise
        """
        return self._is_pointer_type

    @property
    def vector_length(self):
        if self._vector_length is None:
            return 0
        return self._vector_length

    def set_address_space_qualifier(self, address_space_qualifier):
        """Set the address space qualifier.

        Args:
            address_space_qualifier (str): the new address space qualifier

        Returns:
            self: for chaining
        """
        self.address_space_qualifier = address_space_qualifier
        return self

    def set_pre_data_type_type_qualifiers(self, pre_data_type_type_qualifiers):
        """Set the pre data type type qualifier.

        Args:
            pre_data_type_type_qualifiers (list of str): the pre data type type qualifiers

        Returns:
            self: for chaining
        """
        self.pre_data_type_type_qualifiers = pre_data_type_type_qualifiers
        return self

    def set_post_data_type_type_qualifier(self, post_data_type_type_qualifier):
        """Set the post data type type qualifier.

        Args:
            post_data_type_type_qualifier (str): the post data type type qualifier

        Returns:
            self: for chaining
        """
        self.post_data_type_type_qualifier = post_data_type_type_qualifier
        return self


class AbstractProblemData(object):
    """A simple data container for the data for optimization/sampling models."""

    @property
    def prtcl_data_dict(self):
        """Return the protocol data stored in this problem data container.

        The protocol data contains information about the experimental setup. In MRI this is the scanner protocol.

        Returns:
            dict: The protocol data dict.
        """
        return {}

    @property
    def observations(self):
        """Return the observations stored in this problem data container.

        Returns:
            ndarray: The list of observed instances per problem. Should be a matrix with as columns the observations
                and as rows the problems.
        """
        return np.array([])


class SimpleProblemData(AbstractProblemData):

    def __init__(self, prtcl_data_dict, observations_list):
        """A simple data container for the data for optimization/sampling models.

        Args:
            prtcl_data_dict (dict): The protocol data dictionary
            observations_list (ndarray): The array with the observations
        """
        self._prtcl_data_dict = prtcl_data_dict
        self._observation_list = observations_list

    @property
    def prtcl_data_dict(self):
        return self._prtcl_data_dict

    @property
    def observations(self):
        return self._observation_list


class CLFunction(object):

    def __init__(self, return_type, function_name, parameter_list):
        """The header to one of the CL library functions.

        Ideally the functions map bijective to the CL function files.

        Args:
            return_type (str): Return type of the CL function.
            function_name (string): The name of the CL function
            parameter_list (list of CLFunctionParameter): The list of parameters required for this function
        """
        super(CLFunction, self).__init__()
        self._return_type = return_type
        self._function_name = function_name
        self._parameter_list = parameter_list

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

    def get_cl_header(self):
        """Get the CL header for this function and all its dependencies

        Returns:
            str: The CL header code for inclusion in CL source code.
        """
        return ''

    def get_cl_code(self):
        """Get the function code for this function and all its dependencies.

        Returns:
            str: The CL header code for inclusion in CL source code.
        """
        return ''

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return type(self) != type(other)


class DependentCLFunction(CLFunction):

    def __init__(self, return_type, function_name, parameter_list, dependency_list):
        """A CL function with dependencies on multiple other CLFunctions.

        Args:
            return_type (str): Return type of the CL function.
            function_name (string): The name of the CL function
            parameter_list (list of CLFunctionParameter): The list of parameters required for this function
            dependency_list (list of CLFunction): The list of CLFunctions this function is dependent on
        """
        super(DependentCLFunction, self).__init__(return_type, function_name, parameter_list)
        self._dependency_list = dependency_list

    def _get_cl_dependency_headers(self):
        """Get the CL code for all the headers for all the dependencies.

        Returns:
            str: The CL code with the headers.
        """
        header = ''
        for d in self._dependency_list:
            header += d.get_cl_header() + "\n"
        return header

    def _get_cl_dependency_code(self):
        """Get the CL code for all the CL code for all the dependencies.

        Returns:
            str: The CL code with the actual code.
        """
        code = ''
        for d in self._dependency_list:
            code += d.get_cl_code() + "\n"
        return code


class ModelFunction(DependentCLFunction):

    def __init__(self, name, cl_function_name, parameter_list, dependency_list=()):
        """This CL function is for all estimable models

        Args:
            name (str): The name of the model
            cl_function_name (string): The name of the CL function
            parameter_list (list or tuple of CLFunctionParameter): The list of parameters required for this function
            dependency_list (list or tuple of CLFunction): The list of CLFunctions this function is dependent on
        """
        super(ModelFunction, self).__init__('double', cl_function_name, parameter_list, dependency_list)
        self._name = name

    @property
    def name(self):
        """Get the name of this model function.

        Returns:
            str: The name of this model function.
        """
        return self._name

    def get_free_parameters(self):
        """Get all parameters whose state instance is one of the type CLPSFreeState.

        Returns:
            A list of parameters whose type matches the CLPSFreeState state type.
        """
        return self.get_parameters_of_type(FreeParameter)

    def get_parameters_of_type(self, instance_types):
        """Get all parameters whose state instance is one of the given types.

        Args:
            instance_types (list of DataType class names, or single DataType classname);
                The instance type we want to get all the parameters of.

        Returns:
            A list of parameters whose type matches one or more of the given types.
        """
        return list([p for p in self.parameter_list if isinstance(p, instance_types)])

    def init(self, param_name, value):
        """Set the parameter (by the param_name) to the given value (single value or array)

        This generally only works for parameters with the state CLPSFreeState, for all other states it may fail.

        Args:
            param_name (str): the name of the parameter to initialize
            value (number or ndarray): the optional value to initialize it to
        """
        self.get_parameter_by_name(param_name).value = value
        return self

    def unfix(self, param_name):
        """Unfix this parameter, which means, make sure it is no longer fixed if it was fixed.

        This generally only works for parameters with the state CLPSFreeState, for all other states it may fail.

        Args:
           param_name (str): the name of the parameter
        """
        self.get_parameter_by_name(param_name).unfix()
        return self

    def fix(self, param_name, value=None):
        """Fix this parameter to the given value (single value or array of values)

        This generally only works for parameters with the state CLPSFreeState, for all other states it may fail.

        Args:
            param_name (str): the name of the parameter
            value (number or ndarray): the value to fix it to (one value, or an array with one value per voxel)
                If value is None, the value is fixed to the default value (if possible)
        """
        if value is not None:
            self.get_parameter_by_name(param_name).fix_to(value)
        self.get_parameter_by_name(param_name).fixed = True
        return self

    def ub(self, param_name, value):
        """Set the upper bound for this parameter to the specific value.

        This may be used during parameter transformations and or sampling proposal generation and or the sampling
        prior function.

        This generally only works for parameters with the state CLPSFreeState, for all other states it may fail.

        Args:
            param_name: the name of the parameter
            value: the upper bound
        """
        self.get_parameter_by_name(param_name).upper_bound = value
        return self

    def lb(self, param_name, value):
        """Set the lower bound for this parameter to the specific value.

        This may be used during parameter transformations and or sampling proposal generation and or the sampling
        prior function.

        This generally only works for parameters with the state CLPSFreeState, for all other states it may fail.

        Args:
            param_name: the name of the parameter
            value: the lower bound
        """
        self.get_parameter_by_name(param_name).lower_bound = value
        return self

    def has_parameter_by_name(self, param_name):
        """Check if this model function has a parameter with the given name.

        Args:
            param_name (str): The name of the parameter to check for existence.

        Returns:
            boolean: True if a parameter of the given name is attached to this function, else otherwise.
        """
        try:
            self.get_parameter_by_name(param_name)
            return True
        except KeyError:
            return False

    def p(self, param_name):
        """Get a parameter by name.

        This function is a short for the function get_parameter_by_name()

        Args:
            param_name (str): The name of the parameter to return

        Returns:
            ClFunctionParameter: the parameter of the given name
        """
        return self.get_parameter_by_name(param_name)

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

    def get_extra_results_maps(self, results_dict):
        """Get extra results maps with extra output from this model function.

        This is used by the function finalize_optimization_results() from the ModelBuilder to add extra maps
        to the resulting dictionary.

        Suppose a model has a parameter that can be viewed in multiple ways. It would be nice to be able
        to output maps for that parameter in multiple ways such that the amount of post-processing is as least as
        possible.

        For example, suppose a model calculates an angle (theta) and a radius (r). Perhaps we would like to return
        the cartesian coordinate of that point alongside the polar coordinates. This function allows you (indirectly)
        to add the additional maps.

        Do not modify the dictionary in place.

        Args:
            results_dict (dict): The result dictionary with all the maps you need and perhaps other maps from other
                models as well. The maps are 1 dimensional, a long list of values for all the voxels in the ROI.

        Returns:
            dict: A new dictionary with the additional maps to add.
        """
        return {}


class LibraryFunction(DependentCLFunction):

    def __init__(self, return_type, function_name, parameter_list, cl_header_file,
                 cl_code_file, var_replace_dict, dependency_list):
        """Create a CL function for a library function.

        These functions are not meant to be optimized, but can be used a helper functions for the models.

        Args:
            return_type (str): Return type of the CL function.
            function_name (str): The name of the CL function
            cl_header_file (str): The location of the header file .h or .ph
            cl_code_file (str): The location of the code file .c or .pcl
            var_replace_dict (dict): In the cl_header and cl_code file these replacements will be made
                (using the % format function of Python)
            parameter_list (list or tuple of CLFunctionParameter): The list of parameters required for this function
            dependency_list (list or tuple of CLFunction): The list of CLFunctions this function is dependent on
        """
        super(LibraryFunction, self).__init__(return_type, function_name, parameter_list, dependency_list)
        self._cl_header_file = cl_header_file
        self._cl_code_file = cl_code_file
        self._var_replace_dict = var_replace_dict

    def get_cl_header(self):
        """Get the CL header for this function and all its dependencies.

        Returns:
            str: The CL code for the header
        """
        header = self._get_cl_dependency_headers()
        header += "\n"
        body = open(os.path.abspath(self._cl_header_file), 'r').read()
        if self._var_replace_dict:
            body = body % self._var_replace_dict
        return header + "\n" + body

    def get_cl_code(self):
        """Get the CL code for this function and all its dependencies.

        Returns:
            str: The CL code for the body of the function
        """
        code = self._get_cl_dependency_code()
        code += "\n"
        body = open(os.path.abspath(self._cl_code_file), 'r').read()
        if self._var_replace_dict:
            body = body % self._var_replace_dict
        return code + "\n" + body


class CLFunctionParameter(object):

    def __init__(self, data_type, name):
        """Creates a new function parameter for the CL functions.

        Args:
            data_type (DataType): the data type expected by this parameter
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


class FreeParameter(CLFunctionParameter):

    def __init__(self, data_type, name, fixed, value, lower_bound, upper_bound,
                 parameter_transform=None, sampling_proposal=None,
                 sampling_prior=None, sampling_statistics=None, perturbation_function=None):
        """This are the kind of parameters that are generally meant to be optimized.

        These parameters may optionally be fixed to a value or list of values for all voxels.

        Args:
            data_type (DataType): the data type expected by this parameter
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
            perturbation_function (python cb func): The perturbation function to use for perturbing this parameter
                in between optimization runs. It should accept a single parameter, an array with values to uniquely
                perturb.

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
            perturbation_function (python cb func): The perturbation function to use for perturbing this parameter
                in between optimization runs. It should accept a single parameter, an array with values to uniquely
                perturb.
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
        self.perturbation_function = perturbation_function or (lambda v: v)

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
            data_type (DataType): the data type expected by this parameter
            name (str): The name of this parameter
            value (double or ndarray): A single value for all voxels or a list of values for each voxel

        Attributes:
            value (double or ndarray): A single value for all voxels or a list of values for each voxel
        """
        super(ModelDataParameter, self).__init__(data_type, name)
        self.value = value
