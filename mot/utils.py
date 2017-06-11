import logging
from contextlib import contextmanager
from functools import reduce
import numpy as np
import pyopencl as cl

__author__ = 'Robbert Harms'
__date__ = "2014-05-13"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


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
    return (value == value[0]).all()


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
