import numbers
import numpy as np
from pyopencl import array as cl_array

__author__ = 'Robbert Harms'
__date__ = "2016-12-06"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DataAdapter(object):
    """Create a data adapter for the given data and type.

    This data adapter is the bridge between the raw data and the data used in the kernels.
    """

    def get_data_type(self):
        """Get the data type for the data in this adapter.

        Returns:
            mot.cl_data_type.CLDataType: the datatype
        """
        raise NotImplementedError()

    def get_opencl_data(self):
        """Adapt and return the data for use in OpenCL kernels.

        Returns:
            np.ndarray: the data to be used in compute kernels.
        """
        raise NotImplementedError()

    def get_opencl_numpy_type(self):
        """Get the numpy type for the data in this class for OpenCL use.

        Returns:
            np.dtype: the numpy type for the data
        """
        raise NotImplementedError()

    def allow_local_pointer(self):
        """If this data can be put in a local storage pointer if there is enough memory for it.

        Returns:
            boolean: if we allow this memory to be referenced by a local pointer or not.
        """
        raise NotImplementedError()


class SimpleDataAdapter(DataAdapter):

    def __init__(self, data, data_type, mot_float_type, allow_local_pointer=True):
        """Create a data adapter for the given data and type.

        This data adapter is the bridge between the raw data and the data used in the kernels. If in the future we
        want to add computation types like CUDA or plain C, this adapter knows how to format the data for those targets.

        Args:
            value (ndarray): The value to adapt to different run environments
            data_type (mot.cl_data_type.CLDataType): the data type we need to convert it to
            mot_float_type (mot.cl_data_type.CLDataType): the data type of the mot_float_type
            allow_local_pointer (boolean): if this data can be referenced by a local pointer in the kernel (if
                there is enough memory for it).
        """
        self._data = data
        self._data_type = data_type
        self._mot_float_type = mot_float_type
        self._allow_local_pointer = allow_local_pointer

    def get_opencl_data(self):
        if self._data_type.is_vector_type:
            return self._array_to_cl_vector()
        else:
            return self._get_cl_array()

    def get_opencl_numpy_type(self):
        if self._data_type.is_vector_type:
            return self._get_opencl_vector_data_type()
        else:
            return self._get_cl_numpy_type(self._data_type)

    def get_data_type(self):
        return self._data_type

    def allow_local_pointer(self):
        return self._allow_local_pointer

    def _get_cl_array(self):
        """Convert the data to a numpy array of the current data type.

        Returns:
            np.ndarray: the converted data as a numpy type
        """
        numpy_type = self._get_cl_numpy_type(self._data_type)
        if isinstance(self._data, numbers.Number):
            return numpy_type(self._data)
        return np.require(self._data, numpy_type, ['C', 'A', 'O'])

    def _get_cl_numpy_type(self, data_type):
        """Get the numpy data type for non-vector types in CL.

        This function is not part of the CLDataType class since the numpy datatype may differ depending
        on the final use case.

        Args:
            data_type (mot.cl_data_type.CLDataType): the data type to convert to an numpy type
        """
        raw_type = data_type.raw_data_type

        if raw_type == 'int':
            return np.int32
        if raw_type == 'uint':
            return np.uint32
        if raw_type == 'long':
            return np.int64
        if raw_type == 'ulong':
            return np.uint64
        if raw_type == 'float':
            return np.float32
        if raw_type == 'double':
            return np.float64
        if raw_type == 'mot_float_type':
            return self._get_cl_numpy_type(self._mot_float_type)

    def _array_to_cl_vector(self):
        """Create a CL vector type of the given array.

        Returns:
            ndarray: An array of the same length as the given array, but with only one column per row, the opencl vector.
        """
        s = self._data.shape
        if len(s) > 1:
            width = s[1]
        else:
            width = 1

        dtype = self._get_opencl_vector_data_type()

        ve = np.zeros((s[0], 1), dtype=dtype, order='C')
        for i in range(s[0]):
            for j in range(width):
                ve[i, 0][j] = self._data[i, j]

        return ve

    def _get_opencl_vector_data_type(self):
        """Get the data type for a vector of the given length and given type.

        Returns:
            The vector type given the given vector length and data type
        """
        s = self._data.shape
        if len(s) > 1:
            vector_length = s[1]
        else:
            vector_length = 1

        if self._data_type.raw_data_type == 'double':
            data_type = 'double'
        elif self._data_type.raw_data_type == 'mot_float_type':
            data_type = self._mot_float_type.raw_data_type
        else:
            data_type = 'float'

        if vector_length not in (2, 3, 4, 8, 16):
            raise ValueError('The given vector length is not one of (2, 3, 4, 8, 16)')
        if data_type not in ('char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long',
                             'ulong', 'float', 'double', 'half'):
            raise ValueError('The given data type ({}) is not supported.'.format(data_type))

        return getattr(cl_array.vec, data_type + str(vector_length))
