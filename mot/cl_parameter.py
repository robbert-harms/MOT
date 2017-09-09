from copy import copy
import six
from mot.cl_data_type import SimpleCLDataType

__author__ = 'Robbert Harms'
__date__ = '2017-08-31'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class CLFunctionParameter(object):

    @property
    def data_type(self):
        """Get the CL data type of this parameter

        Returns:
            mot.cl_data_type.SimpleCLDataType: The CL data type.
        """
        raise NotImplementedError()

    @property
    def is_cl_vector_type(self):
        """Parse the data_type to see if this parameter holds a vector type (in CL)

        Returns:
            bool: True if the type of this function parameter is a CL vector type.

            CL vector types are recognized by an integer after the data type. For example: double4 is a
            CL vector type with 4 doubles.
        """
        raise NotImplementedError()

    def get_renamed(self, name):
        """Get a copy of the current parameter but then with a new name.

        Args:
            name (str): the new name for this parameter

        Returns:
            cls: a copy of the current type but with a new name
        """
        raise NotImplementedError()


class SimpleCLFunctionParameter(CLFunctionParameter):

    def __init__(self, data_type, name):
        """Creates a new function parameter for the CL functions.

        Args:
            data_type (mot.cl_data_type.SimpleCLDataType or str): the data type expected by this parameter
                If a string is given we will use ``SimpleCLDataType.from_string`` for translating the data_type.
            name (str): The name of this parameter

        Attributes:
            name (str): The name of this parameter
        """
        if isinstance(data_type, six.string_types):
            self._data_type = SimpleCLDataType.from_string(data_type)
        else:
            self._data_type = data_type

        self.name = name

    @property
    def data_type(self):
        return self._data_type

    @property
    def is_cl_vector_type(self):
        return self._data_type.is_vector_type

    def get_renamed(self, name):
        new_param = copy(self)
        new_param.name = name
        return new_param

