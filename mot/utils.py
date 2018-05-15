import hashlib
import logging
import multiprocessing
import numbers
import os
from collections import Sequence
from contextlib import contextmanager
from functools import reduce

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from mot.cl_data_type import SimpleCLDataType

__author__ = 'Robbert Harms'
__date__ = "2014-05-13"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def add_include_guards(cl_str, guard_name=None):
    """Add include guards to the given string.

    If you are including the same body of CL code multiple times in a Kernel, it is important to add include
    guards (https://en.wikipedia.org/wiki/Include_guard) around them to prevent the kernel from registering the function
    twice.

    Args:
        cl_str (str): the piece of CL code as a string to which we add the include guards
        guard_name (str): the name of the C pre-processor guard. If not given we use the MD5 hash of the
            given cl string.

    Returns:
        str: the same string but then with include guards around them.
    """
    if not guard_name:
        guard_name = 'GUARD_' + hashlib.md5(cl_str.encode('utf-8')).hexdigest()

    return '''
        # ifndef {guard_name}
        # define {guard_name}
        {func_str}
        # endif // {guard_name}
    '''.format(func_str=cl_str, guard_name=guard_name)


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
        shape = data.shape
        dtype = ctype_to_dtype(data_type, mot_float_type.raw_data_type)
        ve = np.zeros(shape[:-1], dtype=dtype)

        if len(shape) == 1:
            for vector_ind in range(shape[0]):
                ve[0][vector_ind] = data[vector_ind]
        elif len(shape) == 2:
            for i in range(data.shape[0]):
                for vector_ind in range(data.shape[1]):
                    ve[i][vector_ind] = data[i, vector_ind]
        elif len(shape) == 3:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    for vector_ind in range(data.shape[2]):
                        ve[i, j][vector_ind] = data[i, j, vector_ind]

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


def get_float_type_def(double_precision):
    """Get the model floating point type definition.

    The MOT_INT_CMP_TYPE is meant for the select() function where you need a long in the case of double precision.

    Args:
        double_precision (boolean): if True we will use the double type for the mot_float_type type.
            Else, we will use the single precision float type for the mot_float_type type.

    Returns:
        str: defines the mot_float_type types, the epsilon and the MIN and MAX values.
    """
    scipy_constants = '''
        #define MACHEP DBL_EPSILON
        #define MAXLOG log(DBL_MAX)
        #define LANCZOS_G 6.024680040776729583740234375 /* taken from Scipy */
        #define EULER 0.577215664901532860606512090082402431 /* Euler constant, from Scipy */
    '''

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
        ''' + scipy_constants
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
        ''' + scipy_constants


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


class NameFunctionTuple(Sequence):

    def __init__(self, name, function):
        self._name = name
        self._function = function

    def get_cl_code(self):
        """Return the CL function.

        Returns:
            str: the CL function
        """
        return self._function

    def get_cl_function_name(self):
        """Get the CL name of this function.

        Returns:
            str: the name of the function
        """
        return self._name

    def __repr__(self):
        return self._function

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (self._name, self._function)[index]


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
    """Split the total number of elements into batches of the specified maximum size.

    Examples::
        split_in_batches(30, 8) -> [(0, 8), (8, 15), (16, 23), (24, 29)]

        for batch_start, batch_end in split_in_batches(2000, 100):
            array[batch_start:batch_end]

    Yields:
        tuple: the start and end point of the next batch
    """
    offset = 0
    elements_left = nmr_elements
    while elements_left > 0:
        next_batch = (offset, offset + min(elements_left, max_batch_size))
        yield next_batch

        batch_size = min(elements_left, max_batch_size)
        elements_left -= batch_size
        offset += batch_size


def hessian_to_covariance(hessian, output_singularity=False):
    """Calculate a covariance matrix from a Hessian by inverting the Hessian.

    Mathematically we can calculate the covariance matrix from the Hessian (the Hessian at the Maximum Likelihood
    Estimator), by a simple matrix inversion. However, round-off errors can make the Hessian singular, making an
    exact inverse impossible. This method uses an exact inverse if possible with a fall back on a pseudo inverse.
    If also the pseudo inverse fails, this function returns zeros as covariance for that Hessian.

    Important: Before the matrix inversion it will set NaN's to 0. After the inversion we make the diagonal
    (representing the variances of each parameter) positive where needed by taking the absolute.

    Args:
        hessian (ndarray): a matrix of shape (n, p, p) where for n problems we have a matrix of shape (p, p) for
            p parameters and we take the inverse for every (p, p) matrix.
        output_singularity (boolean): if set to True, we additionally output a boolean matrix with the location
            of singular Hessians.

    Returns:
        ndarray or tuple: if ``output_singularity`` is set to False, only output the inverse of the Hessians as the
            covariance matrix. If ``output_singularity`` is set to True this function returns a tuple with:
            (``covariance_matrix``, ``is_singular``).
    """
    hessian = np.nan_to_num(hessian)

    if output_singularity:
        is_singular = np.zeros(hessian.shape[0], dtype=bool)

    covars = np.zeros_like(hessian)
    for roi_ind in range(hessian.shape[0]):
        if output_singularity:
            is_singular[roi_ind] = True

        try:
            covars[roi_ind] = np.linalg.inv(hessian[roi_ind])

            if output_singularity:
                is_singular[roi_ind] = False
        except np.linalg.linalg.LinAlgError:
            try:
                covars[roi_ind] = np.linalg.pinv(hessian[roi_ind])
            except np.linalg.linalg.LinAlgError:
                covars[roi_ind] = np.zeros(hessian[roi_ind].shape)

    diagonal_ind = np.arange(hessian.shape[1])
    covars[:, diagonal_ind, diagonal_ind] = np.abs(covars[:, diagonal_ind, diagonal_ind])

    if output_singularity:
        return covars, is_singular
    return covars


def covariance_to_correlations(covariance):
    """Transform a covariance matrix into a correlations matrix.

    This can be seen as dividing a covariance matrix by the outer product of the diagonal.

    As post processing we replace the infinities and the NaNs with zeros and clip the result to [-1, 1].

    Args:
        covariance (ndarray): a matrix of shape (n, p, p) with for n problems the covariance matrix of shape (p, p).

    Returns:
        ndarray: the correlations matrix
    """
    diagonal_ind = np.arange(covariance.shape[1])
    diagonal_els = covariance[:, diagonal_ind, diagonal_ind]
    result = covariance / np.sqrt(diagonal_els[:, :, None] * diagonal_els[:, None, :])
    result[np.isinf(result)] = 0
    return np.clip(np.nan_to_num(result), -1, 1)


class KernelDataManager(object):

    def __init__(self, kernel_input_dict, mot_float_dtype):
        """This class manages the transfer and definitions of the user input data into and from the kernel.

        Args:
            kernel_input_dict (dict[str: KernelData]): the kernel input data items by name
            mot_float_dtype (dtype): a numpy datatype indicating the data type we must use for inputs with ctype
                ``mot_float_type``.
        """
        self._kernel_input_dict = kernel_input_dict or []
        self._input_order = list(sorted(self._kernel_input_dict))
        self._data_duplicates = self.get_duplicate_items()
        self._mot_float_dtype = mot_float_dtype

        for input_data in kernel_input_dict.values():
            input_data.set_mot_float_dtype(mot_float_dtype)

    def get_duplicate_items(self):
        """Get a list of duplicate kernel inputs such that we don't load duplicate information twice.

        This will search for duplicate elements that are to be loaded in the kernel call, point to exactly the
        same memory and are defined to be write only.

        Returns:
            dict: a mapping from duplicate inputs (by name) to the only copy of the duplicate input object we load.
        """
        duplicates = {}

        for index in range(len(self._input_order)):
            key = self._input_order[index]
            kernel_input = self._kernel_input_dict[key]
            data = kernel_input.get_data()

            if not kernel_input.include_in_kernel_call() or kernel_input.read_data_back:
                continue

            if key not in duplicates:
                for other_index in range(index + 1, len(self._input_order)):
                    other_key = self._input_order[other_index]
                    other_kernel_input = self._kernel_input_dict[other_key]
                    other_data = other_kernel_input.get_data()

                    if not other_kernel_input.include_in_kernel_call() or other_kernel_input.read_data_back:
                        continue

                    if np.array_equal(data, other_data):
                        duplicates[other_key] = key

        return duplicates

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
            kernel_data = self._kernel_input_dict[name]
            definitions.append(kernel_data.get_struct_declaration(name))

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
            kernel_data = self._kernel_input_dict[name]
            if kernel_data.include_in_kernel_call() and name not in self._data_duplicates:
                definitions.append(kernel_data.get_kernel_argument_declaration(name))
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

            if name in self._data_duplicates:
                name = self._data_duplicates[name]

            kernel_data = self._kernel_input_dict[name]
            definitions.append(kernel_data.get_struct_init_string(name, problem_id_substitute))

        return '{' + ', '.join(definitions) + '}'

    def get_kernel_inputs(self, cl_context, workgroup_size):
        """Get the kernel inputs to load.

        Args:
            cl_context (pyopencl.Context): the context in which we create the buffer
            workgroup_size (int or None): the work group size the kernel will use. If None we are not using any
                local reduction.

        Returns:
            list of kernel input elements (buffers, local memory object, scalars, etc.)
        """
        kernel_inputs = []
        for name in self._input_order:
            data = self._kernel_input_dict[name]
            if data.include_in_kernel_call() and name not in self._data_duplicates:
                kernel_inputs.append(data.get_kernel_inputs(cl_context, workgroup_size))
        return kernel_inputs

    def get_scalar_arg_dtypes(self):
        """Get the location and types of the input scalars.

        Returns:
            list: for every kernel input element either None if the data is a buffer or the numpy data type if
                if is a scalar.
        """
        dtypes = []
        for ind, name in enumerate(self._input_order):
            data = self._kernel_input_dict[name]
            if data.include_in_kernel_call() and name not in self._data_duplicates:
                dtypes.append(data.get_scalar_arg_dtype())
        return dtypes

    def get_items_to_write_out(self):
        """Get the data name and buffer index of the items to write out after kernel execution.

        Returns:
            list: a list with (buffer index, name) tuples where the name refers to the name of the kernel input element
                and the index is the generated buffer index of that item.
        """
        items = []
        input_index = 0
        for name in self._input_order:
            if self._kernel_input_dict[name].read_data_back:
                items.append([input_index, name])
            if self._kernel_input_dict[name].include_in_kernel_call() and name not in self._data_duplicates:
                input_index += 1
        return items


def multiprocess_mapping(func, iterable):
    """Multiprocess mapping the given function on the given iterable.

    This only works in Linux and Mac systems since Windows has no forking capability. On Windows we fall back on
    single processing. Also, if we reach memory limits we fall back on single cpu processing.

    Args:
        func (func): the function to apply
        iterable (iterable): the iterable with the elements we want to apply the function on
    """
    if os.name == 'nt': # In Windows there is no fork.
        return list(map(func, iterable))
    try:
        p = multiprocessing.Pool()
        return_data = list(p.imap(func, iterable))
        p.close()
        p.join()
        return return_data
    except OSError:
        return list(map(func, iterable))

