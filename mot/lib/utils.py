import hashlib
import logging
import multiprocessing
import numbers
import os
from contextlib import contextmanager
from functools import reduce
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import tatsu
from pkg_resources import resource_filename

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
    if is_vector_ctype(cl_type):
        raw_type, vector_length = split_vector_ctype(cl_type)

        if raw_type == 'mot_float_type':
            if is_vector_ctype(mot_float_type):
                raw_type, _ = split_vector_ctype(mot_float_type)
            else:
                raw_type = mot_float_type

        vector_type = raw_type + str(vector_length)
        return getattr(cl_array.vec, vector_type)
    else:
        if cl_type == 'mot_float_type':
            cl_type = mot_float_type

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
            if ctype == cl_type:
                return dtype


def convert_data_to_dtype(data, data_type, mot_float_type='float'):
    """Convert the given input data to the correct numpy type.

    Args:
        data (ndarray): The value to convert to the correct numpy type
        data_type (str): the data type we need to convert the data to
        mot_float_type (str): the data type of the current ``mot_float_type``

    Returns:
        ndarray: the input data but then converted to the desired numpy data type
    """
    scalar_dtype = ctype_to_dtype(data_type, mot_float_type)

    if isinstance(data, numbers.Number):
        data = scalar_dtype(data)

    if is_vector_ctype(data_type):
        shape = data.shape
        dtype = ctype_to_dtype(data_type, mot_float_type)
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


def split_vector_ctype(ctype):
    """Split a vector ctype into a raw ctype and the vector length.

    If the given ctype is not a vector type, we raise an error. I

    Args:
         ctype (str): the ctype to possibly split into a raw ctype and the vector length

    Returns:
        tuple: the raw ctype and the vector length
    """
    if not is_vector_ctype(ctype):
        raise ValueError('The given ctype is not a vector type.')
    for vector_length in [2, 3, 4, 8, 16]:
        if ctype.endswith(str(vector_length)):
            vector_str_len = len(str(vector_length))
            return ctype[:-vector_str_len], int(ctype[-vector_str_len:])


def is_vector_ctype(ctype):
    """Test if the given ctype is a vector type. That is, if it ends with 2, 3, 4, 8 or 16.

    Args:
        ctype (str): the ctype to test if it is an OpenCL vector type

    Returns:
        bool: if it is a vector type or not
    """
    return any(ctype.endswith(str(i)) for i in [2, 3, 4, 8, 16])


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
    dev_extensions = cl_device.extensions.strip().split(' ')
    return 'cl_khr_fp64' in dev_extensions


def get_float_type_def(double_precision, include_complex=True):
    """Get the model floating point type definition.

    Args:
        double_precision (boolean): if True we will use the double type for the mot_float_type type.
            Else, we will use the single precision float type for the mot_float_type type.
        include_complex (boolean): if we include support for complex numbers

    Returns:
        str: defines the mot_float_type types, the epsilon and the MIN and MAX values.
    """
    if include_complex:
        with open(os.path.abspath(resource_filename('mot', 'data/opencl/complex.h')), 'r') as f:
            complex_number_support = f.read()
    else:
        complex_number_support = ''

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

            #define PYOPENCL_DEFINE_CDOUBLE

            typedef double mot_float_type;
            typedef double2 mot_float_type2;
            typedef double4 mot_float_type4;
            typedef double8 mot_float_type8;
            typedef double16 mot_float_type16;

            #define MOT_EPSILON DBL_EPSILON
            #define MOT_MIN DBL_MIN
            #define MOT_MAX DBL_MAX
        ''' + scipy_constants + complex_number_support
    else:
        return '''
            #if __OPENCL_VERSION__ <= CL_VERSION_1_1
                #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            #endif

            typedef float mot_float_type;
            typedef float2 mot_float_type2;
            typedef float4 mot_float_type4;
            typedef float8 mot_float_type8;
            typedef float16 mot_float_type16;

            #define MOT_EPSILON FLT_EPSILON
            #define MOT_MIN FLT_MIN
            #define MOT_MAX FLT_MAX
        ''' + scipy_constants + complex_number_support


def get_atomic_functions(mot_float_type_is_double):
    """Add a few additional atomic functions to all kernels.

    Copied from: https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/

    Todo: remove these when we support OpenCL 2.0

    Args:
        mot_float_type_is_double (bool): if the mot_float_type is double or not
    """
    atomics = '''
        #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

        void atomic_add_g_f(volatile __global float *addr, float val){
            union {
                unsigned int u32;
                float        f32;
            } next, expected, current;

            current.f32    = *addr;
            do {
                expected.f32 = current.f32;
                next.f32     = expected.f32 + val;
                current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u32, next.u32);
            } while( current.u32 != expected.u32 );
        }

        void atomic_add_l_f(volatile __local float *addr, float val){
            union {
                unsigned int u32;
                float        f32;
            } next, expected, current;

            current.f32    = *addr;
            do {
                expected.f32 = current.f32;
                next.f32     = expected.f32 + val;
                current.u32  = atomic_cmpxchg( (volatile __local unsigned int *)addr, expected.u32, next.u32);
            } while( current.u32 != expected.u32 );
        }

        void atomic_add_g_d(volatile __global double *addr, double val) {
            union {
                ulong  u64;
                double f64;
            } next, expected, current;

            current.f64 = *addr;
            do {
                expected.f64 = current.f64;
                next.f64     = expected.f64 + val;
                current.u64  = atom_cmpxchg( (volatile __global ulong *)addr, expected.u64, next.u64);
            } while( current.u64 != expected.u64 );
        }

        void atomic_add_l_d(volatile __local double *addr, double val) {
            union {
                ulong  u64;
                double f64;
            } next, expected, current;

            current.f64 = *addr;
            do {
                expected.f64 = current.f64;
                next.f64     = expected.f64 + val;
                current.u64  = atom_cmpxchg( (volatile __local ulong *)addr, expected.u64, next.u64);
            } while( current.u64 != expected.u64 );
        }
    '''
    if mot_float_type_is_double:
        atomics += '''
            void atomic_add_g_mft(volatile __global double *addr, double val){
                atomic_add_g_d(addr, val);
            }

            void atomic_add_l_mft(volatile __local double *addr, double val){
                atomic_add_l_d(addr, val);
            }
        '''
    else:
        atomics += '''
            void atomic_add_g_mft(volatile __global float *addr, float val){
                atomic_add_g_f(addr, val);
            }

            void atomic_add_l_mft(volatile __local float *addr, float val){
                atomic_add_l_f(addr, val);
            }
        '''

    return atomics


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


def multiprocess_mapping(func, iterable):
    """Multiprocess mapping the given function on the given iterable.

    This only works in Linux and Mac systems since Windows has no forking capability. On Windows we fall back on
    single processing. Also, if we reach memory limits we fall back on single cpu processing.

    Args:
        func (func): the function to apply
        iterable (iterable): the iterable with the elements we want to apply the function on
    """
    if os.name == 'nt':  # In Windows there is no fork.
        return list(map(func, iterable))
    try:
        p = multiprocessing.Pool()
        return_data = list(p.imap(func, iterable))
        p.close()
        p.join()
        return return_data
    except OSError:
        return list(map(func, iterable))


_tatsu_cl_function = '''
    function = {documentation}* [address_space] data_type function_name arglist body;
    documentation = '/*' ->'*/';
    address_space = ['__'] ('local' | 'global' | 'constant' | 'private');
    data_type = /\w+(\s*(\*)?)+/;
    function_name = /\w+/;
    arglist = '(' @+:arg {',' @+:arg}* ')' | '()';
    arg = /[\w \*\[\]]+/;
    body = compound_statement;
    compound_statement = '{' {[/[^\{\}]*/] [compound_statement]}* '}';
'''

_extract_cl_functions_parser = tatsu.compile('''
    result = {function}+;
''' + _tatsu_cl_function)

_split_cl_function_parser = tatsu.compile('''
    result = function;
''' + _tatsu_cl_function)


def parse_cl_function(cl_code, dependencies=()):
    """Parse the given OpenCL string to a single SimpleCLFunction.

    If the string contains more than one function, we will return only the last, with all the other added as a
    dependency.

    Args:
        cl_code (str): the input string containing one or more functions.
        dependencies (Iterable[CLCodeObject]): The list of CL libraries this function depends on

    Returns:
        mot.lib.cl_function.SimpleCLFunction: the CL function for the last function in the given strings.
    """
    from mot.lib.cl_function import SimpleCLFunction

    def separate_cl_functions(input_str):
        """Separate all the OpenCL functions.

        This creates a list of strings, with for each function found the OpenCL code.

        Args:
            input_str (str): the string containing one or more functions.

        Returns:
            list: a list of strings, with one string per found CL function.
        """
        class Semantics:

            def __init__(self):
                self._functions = []

            def result(self, ast):
                return self._functions

            def arglist(self, ast):
                return '({})'.format(', '.join(ast))

            def function(self, ast):
                def join(items):
                    result = ''
                    for item in items:
                        if isinstance(item, str):
                            result += item
                        else:
                            result += join(item)
                    return result

                self._functions.append(join(ast).strip())
                return ast

        return _extract_cl_functions_parser.parse(input_str, semantics=Semantics())

    functions = separate_cl_functions(cl_code)
    return SimpleCLFunction.from_string(functions[-1], dependencies=list(dependencies or []) + [
        SimpleCLFunction.from_string(s) for s in functions[:-1]])


def split_cl_function(cl_str):
    """Split an CL function into a return type, function name, parameters list and the body.

    Args:
        cl_str (str): the CL code to parse and plit into components

    Returns:
        tuple: string elements for the return type, function name, parameter list and the body
    """
    class Semantics:

        def __init__(self):
            self._return_type = ''
            self._function_name = ''
            self._parameter_list = []
            self._cl_body = ''

        def result(self, ast):
            return self._return_type, self._function_name, self._parameter_list, self._cl_body

        def address_space(self, ast):
            self._return_type = ast.strip() + ' '
            return ast

        def data_type(self, ast):
            self._return_type += ''.join(ast).strip()
            return ast

        def function_name(self, ast):
            self._function_name = ast.strip()
            return ast

        def arglist(self, ast):
            if ast != '()':
                self._parameter_list = ast
            return ast

        def body(self, ast):
            def join(items):
                result = ''
                for item in items:
                    if isinstance(item, str):
                        result += item
                    else:
                        result += join(item)
                return result

            self._cl_body = join(ast).strip()[1:-1]
            return ast

    return _split_cl_function_parser.parse(cl_str, semantics=Semantics())

