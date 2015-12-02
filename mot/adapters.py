import numbers
import numpy as np
from mot.utils import get_opencl_vector_data_type

__author__ = 'Robbert Harms'
__date__ = "2015-12-02"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class DataAdapter(object):

    def __init__(self, data, data_type, mot_float_type):
        """Create a data adapter for the given data and type.

        Args:
            value (ndarray): The value to adapt to different run environments
            data_type (DataType): the data type we need to convert it to
            mot_float_type (DataType): the datatype of MOT_FLOAT_TYPE
        """
        self._data = data
        self._data_type = data_type
        self._mot_float_type = mot_float_type

    def adapt_to_opencl(self):
        """Adapt and return the data for use in OpenCL kernels.

        Returns:
            data: the data to be used in compute kernels.
        """
        if self._data_type.is_vector_type:
            return self._array_to_cl_vector()
        else:
            return self._get_cl_array()

    def _get_cl_array(self):
        """Convert the data to a numpy array of the current data type

        Returns:
            np ndarray: the converted self._data as a numpy type
        """
        numpy_type = self._get_cl_numpy_type(self._data_type)
        if isinstance(self._data, numbers.Number):
            return numpy_type(self._data)
        return self._data.astype(numpy_type, copy=False, order='C')

    def _get_cl_numpy_type(self, data_type):
        """Get the data type for non-vector types in CL."""
        raw_type = data_type.raw_data_type

        if raw_type == 'float':
            return np.float32
        if raw_type == 'int':
            return np.int32
        if raw_type == 'double':
            return np.float64
        if raw_type == 'MOT_FLOAT_TYPE':
            return self._get_cl_numpy_type(self._mot_float_type)

    def _array_to_cl_vector(self):
        """Create a CL vector type of the given array.

        Returns:
            ndarray: An array of the same length as the given array, but with only one column per row.
                This column contains the opencl vector.
        """
        s = self._data.shape
        if len(s) > 1:
            width = s[1]
        else:
            width = 1

        vector_length = width

        if self._data_type.raw_data_type == 'double':
            dtype = get_opencl_vector_data_type(vector_length, 'double')
        elif self._data_type.raw_data_type == 'MOT_FLOAT_TYPE':
            dtype = get_opencl_vector_data_type(vector_length, self._mot_float_type.raw_data_type)
        else:
            dtype = get_opencl_vector_data_type(vector_length, 'float')

        ve = np.zeros((s[0], 1), dtype=dtype, order='C')
        for i in range(s[0]):
            for j in range(width):
                ve[i, 0][j] = self._data[i, j]

        return ve
