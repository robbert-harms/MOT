import logging
from contextlib import contextmanager
from functools import reduce
import numpy as np
import pyopencl as cl
import numbers
from mot.cl_data_type import SimpleCLDataType
import pyopencl.array as cl_array

__author__ = 'Robbert Harms'
__date__ = "2014-05-13"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def dtype_to_ctype(dtype):
    """Get the CL type of the given numpy data type.

    Args:
        dtype (np.dtype): the numpy data type

    Returns:
        str: the CL type string for the corresponding type
    """
    from pyopencl.tools import dtype_to_ctype
    return dtype_to_ctype(dtype)


def ctype_to_dtype(cl_type, mot_float_type='float'):
    """Get the numpy dtype of the given cl_type string.

    Args:
        cl_type (str): the CL data type to match, for example 'float' or 'float4'.
        mot_float_type (str): the C name of the ``mot_float_type``. The dtype will be looked up recursively.

    Returns:
        dtype: the numpy datatype
    """
    data_type = SimpleCLDataType.from_string(cl_type)

    if data_type.is_vector_type:
        if data_type.raw_data_type.startswith('mot_float_type'):
            data_type = SimpleCLDataType.from_string(mot_float_type + str(data_type.vector_length))
        vector_type = data_type.raw_data_type + str(data_type.vector_length)
        return getattr(cl_array.vec, vector_type)
    else:
        if data_type.raw_data_type.startswith('mot_float_type'):
            data_type = SimpleCLDataType.from_string(mot_float_type)
        data_types = [
            ('char', np.int8),
            ('uchar', np.uint8),
            ('short', np.int16),
            ('ushort', np.uint16),
            ('int', np.int32),
            ('uint', np.uint32),
            ('long', np.int64),
            ('ulong', np.uint64),
            ('float', np.float32),
            ('double', np.float64),
        ]
        for ctype, dtype in data_types:
            if ctype == data_type.raw_data_type:
                return dtype


def convert_data_to_dtype(data, data_type, mot_float_type='float'):
    """Convert the given input data to the correct numpy type.

    Args:
        value (ndarray): The value to convert to the correct numpy type
        data_type (str or mot.cl_data_type.CLDataType): the data type we need to convert the data to
        mot_float_type (str or mot.cl_data_type.CLDataType): the data type of the current ``mot_float_type``

    Returns:
        ndarray: the input data but then converted to the desired numpy data type
    """
    if isinstance(data_type, str):
        data_type = SimpleCLDataType.from_string(data_type)
    if isinstance(mot_float_type, str):
        mot_float_type = SimpleCLDataType.from_string(mot_float_type)

    scalar_dtype = ctype_to_dtype(data_type.raw_data_type, mot_float_type.raw_data_type)

    if isinstance(data, numbers.Number):
        data = scalar_dtype(data)

    if data_type.is_vector_type:
        if len(data.shape) < 2:
            data = data[..., None]

        dtype = ctype_to_dtype(data_type, mot_float_type.raw_data_type)

        ve = np.zeros((data.shape[0], 1), dtype=dtype, order='C')
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ve[i, 0][j] = data[i, j]
        return np.require(ve, requirements=['C', 'A', 'O'])
    return np.require(data, scalar_dtype, ['C', 'A', 'O'])


def device_type_from_string(cl_device_type_str):
    """Converts values like ``gpu`` to a pyopencl device type string.

    Supported values are: ``accelerator``, ``cpu``, ``custom``, ``gpu``. If ``all`` is given, None is returned.

    Args:
        cl_device_type_str (str): The string we want to convert to a device type.

    Returns:
        cl.device_type: the pyopencl device type.
    """
    cl_device_type_str = cl_device_type_str.upper()
    if hasattr(cl.device_type, cl_device_type_str):
        return getattr(cl.device_type, cl_device_type_str)
    return None


def device_supports_double(cl_device):
    """Check if the given CL device supports double

    Args:
        cl_device (pyopencl cl device): The device to check if it supports double.

    Returns:
        boolean: True if the given cl_device supports double, false otherwise.
    """
    return cl_device.get_info(cl.device_info.DOUBLE_FP_CONFIG) == 63


def results_to_dict(results, param_names):
    """Create a dictionary out of the results.

    This basically splits the given nd-matrix into sub matrices based on the second dimension. The length of
    the parameter names should match the length of the second dimension. If a two dimensional matrix of shape (d, p) is
    given we return a matrix of shape (d,). If a matrix of shape (d, p, s_1, s_2, ..., s_n) is given, we return
    a matrix of shape (d, s_1, s_2, ..., s_n).

    Args:
        results: a multidimensional matrix we index based on the second dimension.
        param_names (list of str): the names of the parameters, one per column

    Returns:
        dict: the results packed in a dictionary
    """
    if results.shape[1] != len(param_names):
        raise ValueError('The number of columns ({}) in the matrix does not match '
                         'the number of dictionary keys provided ({}).'.format(results.shape[1], len(param_names)))
    return {name: results[:, i, ...] for i, name in enumerate(param_names)}


def get_float_type_def(double_precision):
    """Get the model floating point type definition.

    The MOT_INT_CMP_TYPE is meant for the select() function where you need a long in the case of double precision.

    Args:
        double_precision (boolean): if True we will use the double type for the mot_float_type type.
            Else, we will use the single precision float type for the mot_float_type type.

    Returns:
        str: defines the mot_float_type types, the epsilon and the MIN and MAX values.
    """
    if double_precision:
        return '''
            #if __OPENCL_VERSION__ <= CL_VERSION_1_1
                #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            #endif

            #define mot_float_type double
            #define mot_float_type2 double2
            #define mot_float_type4 double4
            #define mot_float_type8 double8
            #define mot_float_type16 double16
            #define MOT_EPSILON DBL_EPSILON
            #define MOT_MIN DBL_MIN
            #define MOT_MAX DBL_MAX
            #define MOT_INT_CMP_TYPE long
        '''
    else:
        return '''
            #if __OPENCL_VERSION__ <= CL_VERSION_1_1
                #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            #endif

            #define mot_float_type float
            #define mot_float_type2 float2
            #define mot_float_type4 float4
            #define mot_float_type8 float8
            #define mot_float_type16 float16
            #define MOT_EPSILON FLT_EPSILON
            #define MOT_MIN FLT_MIN
            #define MOT_MAX FLT_MAX
            #define MOT_INT_CMP_TYPE int
        '''


def topological_sort(data):
    """Topological sort the given dictionary structure.

    Args:
        data (dict); dictionary structure where the value is a list of dependencies for that given key.
            For example: ``{'a': (), 'b': ('a',)}``, where ``a`` depends on nothing and ``b`` depends on ``a``.

    Returns:
        tuple: the dependencies in constructor order
    """

    def check_self_dependencies(input_data):
        """Check if there are self dependencies within a node.

        Self dependencies are for example: ``{'a': ('a',)}``.

        Args:
            input_data (dict): the input data. Of a structure similar to {key: (list of values), ...}.

        Raises:
            ValueError: if there are indeed self dependencies
        """
        for k, v in input_data.items():
            if k in v:
                raise ValueError('Self-dependency, {} depends on itself.'.format(k))

    def prepare_input_data(input_data):
        """Prepares the input data by making sets of the dependencies. This automatically removes redundant items.

        Args:
            input_data (dict): the input data. Of a structure similar to {key: (list of values), ...}.

        Returns:
            dict: a copy of the input dict but with sets instead of lists for the dependencies.
        """
        return {k: set(v) for k, v in input_data.items()}

    def find_items_without_dependencies(input_data):
        """This searches the dependencies of all the items for items that have no dependencies.

        For example, suppose the input is: ``{'a': ('b',)}``, then ``a`` depends on ``b`` and ``b`` depends on nothing.
        This class returns ``(b,)`` in this example.

        Args:
            input_data (dict): the input data. Of a structure similar to {key: (list of values), ...}.

        Returns:
            list: the list of items without any dependency.
        """
        return list(reduce(set.union, input_data.values()) - set(input_data.keys()))

    def add_empty_dependencies(data):
        items_without_dependencies = find_items_without_dependencies(data)
        data.update({item: set() for item in items_without_dependencies})

    def get_sorted(input_data):
        data = input_data
        while True:
            ordered = set(item for item, dep in data.items() if len(dep) == 0)
            if not ordered:
                break
            yield ordered
            data = {item: (dep - ordered) for item, dep in data.items() if item not in ordered}

        if len(data) != 0:
            raise ValueError('Cyclic dependencies exist '
                             'among these items: {}'.format(', '.join(repr(x) for x in data.items())))

    check_self_dependencies(data)

    if not len(data):
        return []

    data_copy = prepare_input_data(data)
    add_empty_dependencies(data_copy)

    result = []
    for d in get_sorted(data_copy):
        try:
            d = sorted(d)
        except TypeError:
            d = list(d)

        result.extend(d)
    return result


def is_scalar(value):
    """Test if the given value is a scalar.

    This function also works with memory mapped array values, in contrast to the numpy is_scalar method.

    Args:
        value: the value to test for being a scalar value

    Returns:
        boolean: if the given value is a scalar or not
    """
    return np.isscalar(value) or (isinstance(value, np.ndarray) and (len(np.squeeze(value).shape) == 0))


def all_elements_equal(value):
    """Checks if all elements in the given value are equal to each other.

    If the input is a single value the result is trivial. If not, we compare all the values to see
    if they are exactly the same.

    Args:
        value (ndarray or number): a numpy array or a single number.

    Returns:
        bool: true if all elements are equal to each other, false otherwise
    """
    if is_scalar(value):
        return True
    return np.array(value == value.flatten()[0]).all()


def get_single_value(value):
    """Get a single value out of the given value.

    This is meant to be used after a call to :func:`all_elements_equal` that returned True. With this
    function we return a single number from the input value.

    Args:
        value (ndarray or number): a numpy array or a single number.

    Returns:
        number: a single number from the input

    Raises:
        ValueError: if not all elements are equal
    """
    if not all_elements_equal(value):
        raise ValueError('Not all values are equal to each other.')

    if is_scalar(value):
        return value
    return value.item(0)


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """Disable all logging temporarily.

    A context manager that will prevent any logging messages triggered during the body from being processed.

    Args:
        highest_level: the maximum logging level that is being blocked
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)


class NamedCLFunction(object):

    def get_cl_code(self):
        """Return the CL function.

        Returns:
            str: the CL function
        """
        raise NotImplementedError()

    def get_cl_function_name(self):
        """Get the CL name of this function.

        Returns:
            str: the name of the function
        """
        raise NotImplementedError()


class SimpleNamedCLFunction(NamedCLFunction):

    def __init__(self, function, name):
        self._function = function
        self._name = name

    def get_cl_code(self):
        return self._function

    def get_cl_function_name(self):
        return self._name

    def __repr__(self):
        return self._function


def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Args:
        arrays (list of array-like): 1-D arrays to form the cartesian product of.
        out (ndarray): Array to place the cartesian product in.

    Returns:
        ndarray: 2-D array of shape (M, len(arrays)) containing cartesian products formed of input arrays.

    Examples:
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
               [1, 4, 7],
               [1, 5, 6],
               [1, 5, 7],
               [2, 4, 6],
               [2, 4, 7],
               [2, 5, 6],
               [2, 5, 7],
               [3, 4, 6],
               [3, 4, 7],
               [3, 5, 6],
               [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    nmr_elements = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([nmr_elements, len(arrays)], dtype=dtype)

    m = nmr_elements // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def split_in_batches(nmr_elements, max_batch_size):
    """Split the total number of elements into batches of the specified maximum size or smaller.

    Examples:
        split_in_batches(30, 8) -> [8, 8, 8, 6]

    Returns:
        list: the list of batch sizes
    """
    batch_sizes = [max_batch_size] * (nmr_elements // max_batch_size)
    if nmr_elements % max_batch_size > 0:
        batch_sizes.append(nmr_elements % max_batch_size)
    return batch_sizes


class KernelInputData(object):
    """This class holds the information necessary to load the data into a CL kernel."""

    def get_data(self):
        """Get the underlying data.

        This should return the data such that it can be loaded by PyOpenCL in a kernel. This means that it
        should return the data pre-formatted with the numpy require method with the requirements 'C', 'A', 'O'
        and 'W' if the data is supposed to be writable.

        Returns:
            ndarray: the underlying data object, make sure this is of your desired type.
        """
        raise NotImplementedError()

    def is_writable(self):
        """If this kernel input data must be loaded as a read-write dataset or as read only.

        If this returns true the kernel function must ensure that the data is loaded with read-write permissions.
        This flag will also ensure that the data will be read back from the device after kernel execution, overwriting
        the current data.

        Returns:
            boolean: if this data must be made writable and be read back after function execution.
        """
        raise NotImplementedError()

    def as_pointer(self):
        """Will this value be provided to as a pointer or as a value.

        If this is set to True we will provide the dataset as a pointer at the specified offset to the
        implementing function. If set to False, we will dereference the pointer at the specified offset and provide
        the values to the implementing function (for example the ``mot_data_struct``).

        Returns:
            boolean: if we dereference the pointer or not
        """
        raise NotImplementedError()

    def get_offset_str(self):
        """Get the offset to use for this dataset in the kernel.

        This should return a string that can compute the offset for this dataset. Since the data is loaded into the
        kernel as a 1d array, we need to offset this array to the correct location for every problem instance.
        This offset often includes a scaling with the current problem index, which can be implemented by the kernel
        in various different ways. To apply this scaling one can return a string containing the literal ``{problem_id}``
        which is replaced by the kernel for the correct problem id.

        Returns:
            str: the offset string for offsetting the input array for each problem. Do not add a plus in front
                of the offset, it is implicit.
        """
        raise NotImplementedError()


class SimpleKernelInputData(KernelInputData):

    def __init__(self, data, offset_str=None, as_pointer=True, is_writable=False):
        """A simple implementation of the kernel input data.

        By default, this will try to offset the data in the kernel by the stride of the first dimension multiplied
        with the problem id by the kernel. For example, if a (n, m) matrix is provided, this will offset the data
        by ``{problem_id} * m``.

        By default we will load the data as a pointer instead of a dereferenced value.

        Args:
            data (ndarray): the data to load in the kernel
            offset_str (str): the offset definition, can use ``{problem_id}`` for multiplication purposes. Set to 0
                for no offset.
            as_pointer (boolean): if we want to load this data as a pointer or not
            is_writable (boolean): if the data must be loaded writable or not, defaults to False
        """
        requirements = ['C', 'A', 'O']
        if is_writable:
            requirements.append('W')

        self._data = np.require(data, requirements=requirements)
        self._offset_str = offset_str
        self._as_pointer = as_pointer
        self._is_writable = is_writable

        if self._offset_str is None:
            self._offset_str = str(self._data.strides[0] // self._data.itemsize) + ' * {problem_id}'
        else:
            self._offset_str = str(self._offset_str)

    def get_data(self):
        return self._data

    def as_pointer(self):
        return self._as_pointer

    def get_offset_str(self):
        return self._offset_str

    def is_writable(self):
        return self._is_writable


class DataStructManager(object):

    def __init__(self, kernel_input_dict):
        """This class manages the definition and the instantiation of the mot_data_struct from the list of data inputs.

        Please note that throughout this class we use the sorted keys for the kernel parts generation.

        Args:
            kernel_input_dict (dict[str: KernelInputData]): the kernel input data items by name
        """
        self._kernel_input_dict = kernel_input_dict or []

    def get_struct_definition(self):
        """Return the structure definition of the mot_data_struct.

        This will use the sorted keys for looping through the kernel input items.

        Returns:
            str: the CL code for the structure definition
        """
        if not len(self._kernel_input_dict):
            return '''
                typedef struct{
                    constant void* place_holder;
                } mot_data_struct;
            '''

        definitions = []
        for name in sorted(self._kernel_input_dict):
            kernel_input_data = self._kernel_input_dict[name]

            definition = ''
            if kernel_input_data.as_pointer():
                definition += 'global '

            definition += dtype_to_ctype(kernel_input_data.get_data().dtype)

            if kernel_input_data.as_pointer():
                definition += '* '
            else:
                definition += ' '

            definition += name + ';'
            definitions.append(definition)

        return '''
            typedef struct{
                ''' + '\n'.join(definitions) + '''
            } mot_data_struct;
        '''

    def get_kernel_arguments(self):
        """Get the list of kernel arguments for loading the kernel data elements into the kernel.

        This will use the sorted keys for looping through the kernel input items.

        Returns:
            list of str: the list of parameter definitions
        """
        definitions = []
        for name in sorted(self._kernel_input_dict):
            kernel_input_data = self._kernel_input_dict[name]
            definitions.append('global {}* {}'.format(dtype_to_ctype(kernel_input_data.get_data().dtype), name))
        return definitions

    def get_struct_init_string(self, problem_id_substitute):
        """Create the structure initialization string.

        This will use the sorted keys for looping through the kernel input items.

        Args:
            problem_id_substitute (str): the substitute for the ``{problem_id}`` in the kernel data info elements.

        Returns:
            str: the instantiation string for the data struct
        """
        if not len(self._kernel_input_dict):
            return '{0}'

        definitions = []
        for name in sorted(self._kernel_input_dict):
            kernel_input_data = self._kernel_input_dict[name]

            offset = kernel_input_data.get_offset_str().replace('{problem_id}', problem_id_substitute)

            definition = name

            if kernel_input_data.as_pointer():
                definition += ' + ' + offset
            else:
                definition += '[' + offset + ']'
            definitions.append(definition)

        return '{' + ', '.join(definitions) + '}'
