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
    if max_batch_size > nmr_elements:
        return [nmr_elements]

    batch_sizes = [max_batch_size] * (nmr_elements // max_batch_size)
    if nmr_elements % max_batch_size > 0:
        batch_sizes.append(nmr_elements % max_batch_size)
    return batch_sizes


class KernelInputData(object):

    @property
    def is_scalar(self):
        """Check if the implemented input data is a scalar or not.

        Since scalars are loaded differently as buffers in the kernel, we have to check if this data is a scalar or not.

        Returns:
            boolean: if the implemented type should be loaded as a scalar or not
        """
        raise NotImplementedError()

    @property
    def dtype(self):
        """Get the numpy data type of this input data.

        Returns:
             numpy dtype: the numpy data type of this data
        """
        raise NotImplementedError()

    @property
    def read_data_back(self):
        """Check if this input data should be read back after kernel execution.

        Returns:
            boolean: if, after kernel launch, the data should be mapped from the compute device back to host memory.
        """
        raise NotImplementedError()

    def get_data(self):
        """Get the underlying data.

        This should return the current state of the data object.

        Returns:
            the underlying data object, make sure this is of your desired data type. Can return None if
                this input data has no actual data.
        """
        raise NotImplementedError()

    def get_struct_declaration(self, name):
        """Get the declaration of this input data in the ``mot_data_struct`` object.

        Args:
            name (str): the name for this data, i.e. how it is represented in the kernel.

        Returns:
            str: the declaration of this input data in the kernel data struct
        """
        raise NotImplementedError()

    def get_kernel_argument_declaration(self, name):
        """Get the kernel argument declaration of this parameter.

        Args:
            name (str): the name for this data, i.e. how it is represented in the kernel.

        Returns:
            str: the kernel argument declaration
        """
        raise NotImplementedError()

    def get_struct_init_string(self, name, problem_id_substitute):
        """Create the structure initialization string.

        Args:
            name (str): the name of this data in the kernel
            problem_id_substitute (str): the substitute for the ``{problem_id}`` in the kernel data info elements.

        Returns:
            str: the instantiation string for the data struct of this input data
        """
        raise NotImplementedError()

    def get_kernel_inputs(self, cl_context, workgroup_size):
        """Get the kernel CL input object.

        Since the kernels follow the map/unmap paradigm make sure to use the ``USE_HOST_PTR`` when making
        writable data objects.

        Args:
            cl_context (pyopencl.Context): the CL context in which we are working.
            workgroup_size (int): the workgroup size the kernel will use

        Returns:
            a buffer, a local memory object, a scalars, etc., anything that can be loaded into the kernel.
        """
        raise NotImplementedError()


class KernelInputScalar(KernelInputData):

    def __init__(self, value):
        """A kernel input scalar.

        Args:
            value (number): the number to insert into the kernel as a scalar.
        """
        self._value = np.array(value)

    @property
    def is_scalar(self):
        return True

    @property
    def dtype(self):
        return self._value.dtype

    @property
    def read_data_back(self):
        return False

    def get_data(self):
        return self._value

    def get_struct_declaration(self, name):
        return '{} {};'.format(dtype_to_ctype(self._value.dtype), name)

    def get_kernel_argument_declaration(self, name):
        return '{} {}'.format(dtype_to_ctype(self._value.dtype), name)

    def get_struct_init_string(self, name, problem_id_substitute):
        return name

    def get_kernel_inputs(self, cl_context, workgroup_size):
        return self._value


class KernelInputLocalMemory(KernelInputData):

    def __init__(self, dtype, size_func=None):
        """Indicates that a local memory array of the indicated size must be loaded as kernel input data.

        Args:
            dtype (numpy dtype): the data type for this local memory object
            size_func (Function): the function that can calculate the required local memory size (in number of bytes),
                given the workgroup size used by the kernel.
        """
        self._dtype = dtype
        self._size_func = size_func or (lambda workgroup_size: workgroup_size * np.dtype(dtype).itemsize)

    @property
    def is_scalar(self):
        return False

    @property
    def dtype(self):
        return self._dtype

    @property
    def read_data_back(self):
        return False

    def get_data(self):
        return None

    def get_struct_declaration(self, name):
        return 'local {}* {};'.format(dtype_to_ctype(self._dtype), name)

    def get_kernel_argument_declaration(self, name):
        return 'local {}* {}'.format(dtype_to_ctype(self._dtype), name)

    def get_struct_init_string(self, name, problem_id_substitute):
        return name

    def get_kernel_inputs(self, cl_context, workgroup_size):
        return cl.LocalMemory(self._size_func(workgroup_size))


class KernelInputBuffer(KernelInputData):

    def __init__(self, data, offset_str=None, is_writable=False, is_readable=True):
        """Loads the given data as a buffer into the kernel.

        By default, this will try to offset the data in the kernel by the stride of the first dimension multiplied
        with the problem id by the kernel. For example, if a (n, m) matrix is provided, this will offset the data
        by ``{problem_id} * m``.

        This class will pre-formatted the data with the numpy require method such that it can be loaded in the kernel.
        That might change the reference to the data, which is important if data is written back. Never trust your
        reference to the input data and always use :meth:`get_data` for return values.

        Args:
            data (ndarray): the data to load in the kernel
            offset_str (str): the offset definition, can use ``{problem_id}`` for multiplication purposes. Set to 0
                for no offset.
            is_writable (boolean): if the data must be loaded writable or not, defaults to False
            is_readable (boolean): if this data must be made readable
        """
        requirements = ['C', 'A', 'O']
        if is_writable:
            requirements.append('W')

        self._data = np.require(data, requirements=requirements)
        self._offset_str = offset_str
        self._is_writable = is_writable
        self._is_readable = is_readable

        if self._offset_str is None:
            self._offset_str = str(self._data.strides[0] // self._data.itemsize) + ' * {problem_id}'
        else:
            self._offset_str = str(self._offset_str)

    @property
    def is_scalar(self):
        return False

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def read_data_back(self):
        return self._is_writable

    @property
    def is_readable(self):
        """If this kernel input data must be readable by the kernel.

        This is used in conjunction with :meth:`is_writable` when loading the data in the kernel.

        Returns:
            boolean: if this data must be made readable by the kernel function
        """
        return self._is_readable

    @property
    def is_writable(self):
        """Check if this kernel input data will write data back.

        This is used in conjunction with :meth:`is_readable` when loading the data in the kernel.

        If this returns true the kernel function must ensure that the data is loaded with at least write permissions.
        This flag will also ensure that the data will be read back from the device after kernel execution, overwriting
        the current data.

        Returns:
            boolean: if this data must be made writable and be read back after function execution.
        """
        return self._is_writable

    def get_struct_declaration(self, name):
        return 'global {}* {};'.format(dtype_to_ctype(self.get_data().dtype), name)

    def get_kernel_argument_declaration(self, name):
        return 'global {}* {}'.format(dtype_to_ctype(self.get_data().dtype), name)

    def get_struct_init_string(self, name, problem_id_substitute):
        offset = self._offset_str.replace('{problem_id}', problem_id_substitute)
        return name + ' + ' + offset

    def get_data(self):
        return self._data

    def get_kernel_inputs(self, cl_context, workgroup_size):
        if self._is_writable:
            if self._is_readable:
                flags = cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR
            else:
                flags = cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR
        else:
            flags = cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR
        return cl.Buffer(cl_context, flags, hostbuf=self._data)


class KernelInputDataManager(object):

    def __init__(self, kernel_input_dict):
        """This class manages the transfer and definitions of the user input data into and from the kernel.

        Args:
            kernel_input_dict (dict[str: KernelInputData]): the kernel input data items by name
        """
        self._kernel_input_dict = kernel_input_dict or []
        self._input_order = list(sorted(self._kernel_input_dict))

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
        for name in self._input_order:
            kernel_input_data = self._kernel_input_dict[name]
            definitions.append(kernel_input_data.get_struct_declaration(name))

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
        for name in self._input_order:
            kernel_input_data = self._kernel_input_dict[name]
            definitions.append(kernel_input_data.get_kernel_argument_declaration(name))
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
        for name in self._input_order:
            kernel_input_data = self._kernel_input_dict[name]
            definitions.append(kernel_input_data.get_struct_init_string(name, problem_id_substitute))

        return '{' + ', '.join(definitions) + '}'

    def get_kernel_inputs(self, cl_context, workgroup_size):
        """Get the kernel inputs to load.

        Args:
            cl_context (pyopencl.Context): the context in which we create the buffer
            workgroup_size (int): the workgroup size the kernel will use

        Returns:
            list of kernel input elements (buffers, local memory object, scalars, etc.)
        """
        kernel_inputs = []
        for data in [self._kernel_input_dict[key] for key in self._input_order]:
            kernel_inputs.append(data.get_kernel_inputs(cl_context, workgroup_size))
        return kernel_inputs

    def get_scalar_arg_dtypes(self):
        """Get the location and types of the input scalars.

        Returns:
            list: for every kernel input element either None if the data is a buffer or the numpy data type if
                if is a scalar.
        """
        dtypes = [None] * len(self._kernel_input_dict)
        for ind, data in enumerate([self._kernel_input_dict[key] for key in self._input_order]):
            if data.is_scalar:
                dtypes[ind] = data.dtype
        return dtypes

    def get_items_to_write_out(self):
        """Get the data name and buffer index of the items to write out after kernel execution.

        Returns:
            list: a list with (buffer index, name) tuples where the name refers to the name of the kernel input element
                and the index is the generated buffer index of that item.
        """
        items = []
        for ind, name in enumerate(self._input_order):
            if self._kernel_input_dict[name].read_data_back:
                items.append([ind, name])
        return items
