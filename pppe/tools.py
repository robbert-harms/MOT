from collections import defaultdict
import numbers
import pyopencl.array as cl_array
import numpy as np
import re
import pyopencl as cl
import scipy.special
from functools import reduce

__author__ = 'Robbert Harms'
__date__ = "2014-05-13"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


def device_supports_double(cl_device):
    """Check if the given CL device supports double

    Args:
        cl_device (pyopencl cl device): The device to check if it supports double.

    Returns:
        True if the given cl_device supports double, false otherwise.
    """
    return cl_device.get_info(cl.device_info.DOUBLE_FP_CONFIG) == 63


def results_to_dict(results, param_names):
    """Create a dictionary out of the results, which the optimizer can output to the user
            Args:
                - results: a 2d (from optimization) or 3d (from sampling) array that needs to be converted to a
                        dictionary.
                - param_names: the names of the parameters, one per column
            Returns:
                results = the results packed in a dictionary
        """
    s = results.shape
    d = {}
    if len(s) == 2:
        for i in range(len(param_names)):
            d[param_names[i]] = results[:, i]
    else:
        for i in range(len(param_names)):
            d[param_names[i]] = results[:, i, :]
    return d


def set_cl_compatible_data_type(value, data_type):
    """Set the given value (numpy array) to the given data type, one which is CL compatible.

    Args:
        - value: numpy array
        - data type: 'float', 'double' or 'double<int>' or 'float<int>'

    Returns:
        The same array, but then with the correct data type. If the data type includes an integer the value is parsed
        to a CL vector type.
    """
    is_vector = bool(re.compile('\d').search(data_type))
    if is_vector:
        return array_to_cl_vector(value, data_type)
    else:
        s = data_type.replace('*', '')
        s = s.replace(' ', '')

        if s == 'float':
            if isinstance(value, numbers.Number):
                return np.float32(value)
            return value.astype(np.float32)
        elif s == 'int':
            if isinstance(value, numbers.Number):
                return np.int32(value)
            return value.astype(np.int32)
        else:
            if isinstance(value, numbers.Number):
                return np.float64(value)
            return value.astype(np.float64)


def numpy_types_to_cl(data_type):
    """Get the CL type name of the given numpy type. Call this function with argument data.dtype.type."""
    names = {np.float32: 'float',
             np.float64: 'double',
             np.int32: 'int',
             np.int64: 'long',
             np.uint32: 'uint',
             np.uint64: 'ulong'}
    return names[data_type]


def get_opencl_vector_data_type(vector_length, data_type):
    """Get the data type for a vector of the given length and given type.

    Args:
        - vector_length: the length of the CL vector data type
        - data_type: the data/double type of the data
    Returns:
        The vector type given the given vector length and data type
    """
    if vector_length not in (2, 3, 4, 8, 16):
        raise ValueError('The given vector length is not one of (2, 3, 4, 8, 16)')
    if data_type not in ('char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong', 'float', 'double', 'half'):
        raise ValueError('The given data type ({}) is not supported.'.format(data_type))

    return getattr(cl_array.vec, data_type + repr(vector_length))


def array_to_cl_vector(array, data_type, vector_length=None):
    """Create a CL vector type of the given array.

    Standard is chosen for the minimum vector length that can hold the given array, else, if vector_length is specified
    the given length is used, but only if vector length is one of (2, 3, 4, 8, 16).

    Args:
        array: the array of which to translate each row to a vector
        vector_length: if specified (non-None) the desired vector length. It must be one of (2, 3, 4, 8, 16)

    Returns:
        An array of the same length as the given array, but with only one column per row. This column contains the
        opencl vector.

    Raises:
        ValueError: if the vector length is not one of (2, 3, 4, 8, 16)
    """
    s = array.shape
    if len(s) > 1:
        width = s[1]
    else:
        width = 1

    if vector_length is None:
        vector_length = width

    if 'double' in data_type:
        dtype = get_opencl_vector_data_type(vector_length, 'double')
    else:
        dtype = get_opencl_vector_data_type(vector_length, 'float')

    ve = np.zeros((s[0], 1), dtype=dtype, order='C')
    for i in range(s[0]):
        for j in range(width):
            ve[i, 0][j] = array[i, j]

    return ve


def is_cl_vector_type(data):
    """Check if the given numpy array contains a CL vector data type"""
    return len(data.dtype.base.descr) > 1


def vector_type_lookup(data):
    """Return the correct vector type, vector length and type_name ('float' or 'double')

    Args:
        - data: a numpy vector or array of which we want to know the CL data type.

    Returns:
        A dictionary with the keys
            - dtype: the cl data type as a object
            - length: the length of the vector
            - data_type: the type of data, double, float etc.
            - cl_name: <data_type><length>, i.e. the full name of the type for use in CL code

        Returns None if it is an unknown type.
    """
    length = len(data.dtype.base.descr)
    if length < 2:
        return None
    cl_type_name = _numpy_to_cl_dtype_names(data.dtype.fields['x'][0])
    return {'dtype': data.dtype, 'length': length, 'data_type': cl_type_name, 'cl_name': (cl_type_name + repr(length))}


def _numpy_to_cl_dtype_names(cl_data_type_name):
    """Translates the names from numpy types to CL types.

    Example: float64 -> double
    """
    if isinstance(cl_data_type_name, np.dtype):
        cl_data_type_name = cl_data_type_name.name

    m = {'float64': 'double',
         'float32': 'float'}
    if cl_data_type_name in m:
        return m[cl_data_type_name]
    return None


def get_cl_double_extension_definer(platform):
    return '''
        #if defined(cl_khr_fp64)
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #elif defined(cl_amd_fp64)
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        #endif
    '''


def get_bessel_roots(number_of_roots=30):
    """These roots are used in some of the compartment models. It are the roots of the equation J'_1(x) = 0.

    That is, where J_1 is the first order Bessel function of the first kind.
    """
    return scipy.special.jnp_zeros(1, number_of_roots).astype(np.float64)


def create_roi(data, brain_mask):
    """Create and return the region of interest of the given brain volume and mask

    Args:
        data: a brain volume with four dimensions (x, y, z, w) where w is the length of the protocol, or a list
                tuple or dictionary with volumes
        brain_mask: the mask indicating the region of interest, dimensions: (x, y, z)

    Returns:
        Signal lists for each of the given volumes. The axis are: (voxels, protocol)
    """
    def creator(v):
        size = v.shape
        idx = np.transpose(np.nonzero(brain_mask))

        if len(size) < 4:
            signals = np.zeros((idx.shape[0], 1))
            for i in range(idx.shape[0]):
                signals[i, :] = v[idx[i, 0], idx[i, 1], idx[i, 2]]
        else:
            signals = np.zeros((idx.shape[0], size[3]))
            for i in range(idx.shape[0]):
                signals[i, :] = v[idx[i, 0], idx[i, 1], idx[i, 2], :]
        return signals

    if isinstance(data, dict):
        return dict((key, creator(value)) for key, value in data.items())
    elif isinstance(data, list):
        return [creator(value) for value in data]
    elif isinstance(data, tuple):
        return (creator(value) for value in data)
    else:
        return creator(data)


def restore_roi(data, brain_mask, with_volume_ind_dim=True):
    """Restore the given data to a whole brain volume

    The data can be a list, tuple or dictionary or directly a two dimensional list of data points

    Args:
        data: the data as a x dimensional list of voxels, or, a list, tuple, or dict of those voxel lists
        brain_mask: the brain_mask which was used to generate the data list
        with_volume_ind_dim (boolean): If true we return values with 4 dimensions. The extra dimension is for
            the volume index. If false we return 3 dimensions.

    Returns:
        Either a single whole volume, a list, tuple or dict of whole volumes, depending on the given input to data.
        If with_volume_ind_dim is set we return values with 4 dimensions. (x, y, z, 1). If not set we return only
        three dimensions.
    """
    idx = np.transpose(np.nonzero(brain_mask))
    original_shape = brain_mask.shape[0:3]

    def restorer(l, index, shape3d):
        s = l.shape

        if with_volume_ind_dim:
            if len(s) == 1:
                volume = np.zeros(shape3d + (1, ))
            else:
                volume = np.zeros(shape3d + s[1:])
        else:
            volume = np.zeros(shape3d)

        for i in range(index.shape[0]):
            if with_volume_ind_dim:
                if len(s) == 1:
                    volume[index[i, 0], index[i, 1], index[i, 2], :] = l[i]
                else:
                    volume[index[i, 0], index[i, 1], index[i, 2], :] = l[i, :]
            else:
                if len(s) == 1:
                    volume[index[i, 0], index[i, 1], index[i, 2]] = l[i]
                else:
                    volume[index[i, 0], index[i, 1], index[i, 2]] = l[i, :]
        return volume

    if isinstance(data, dict):
        return dict((key, restorer(value, idx, original_shape)) for key, value in data.items())
    elif isinstance(data, list):
        return [restorer(value, idx, original_shape) for value in data]
    elif isinstance(data, tuple):
        return (restorer(value, idx, original_shape) for value in data)
    else:
        return restorer(data, idx, original_shape)


class TopologicalSort(object):

    def __init__(self, data):
        self.data = data

    def get_sorted(self):
        """Topological sort the the given data. The data should consist of a dictionary structure.

        Args:
            data (dict); dictionary structure where the value is a list of dependencies for that given key.
                Example: {'a', (), 'b': ('a',)}, here a depends on nothing and b depends on a.
        """
        if not len(self.data):
            return

        new_data = {}
        for k, v in self.data.items():
            new_v = set([])
            if v:
                new_v = set([e for e in v if e is not k])
            new_data.update({k: new_v})
        data = new_data

        # Find all items that don't depend on anything.
        extra_items_in_deps = reduce(set.union, data.values()) - set(data.keys())

        # Add empty dependences where needed.
        data.update({item: set() for item in extra_items_in_deps})
        while True:
            ordered = set(item for item, dep in data.items() if len(dep) == 0)
            if not ordered:
                break
            yield ordered
            data = {item: (dep - ordered) for item, dep in data.items() if item not in ordered}

        if len(data) != 0:
            raise ValueError('Cyclic dependencies exist among these items: {}'.format(', '.join(repr(x)
                                                                                                for x in data.items())))

    def get_flattened_sort(self, sort=True):
        """Returns a single list of dependencies. For any set returned by
            toposort(), those items are sorted and appended to the result (just to
            make the results deterministic).
        """
        result = []
        for d in self.get_sorted():
            result.extend((sorted if sort else list)(d))
        return result


def get_read_only_cl_mem_flags(cl_environment):
    """Get the right read only flags to use in the given environment."""
    if cl_environment.is_gpu:
        return cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
    else:
        return cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR


def get_read_write_cl_mem_flags(cl_environment):
    """Get the right write only flags to use in the given environment."""
    if cl_environment.is_gpu:
        return cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
    else:
        return cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR


def get_write_only_cl_mem_flags(cl_environment):
    """Get the right write only flags to use in the given environment."""
    if cl_environment.is_gpu:
        return cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR
    else:
        return cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR


def check_array_fits_cl_memory(array, dtype, max_size):
    """Check if the given array when casted to the given type can be fit into the given max_size"""
    return np.product(array.shape) * np.dtype(dtype).itemsize < max_size


def get_cl_data_type_from_data(data):
    """Get the data attributes for the given dataset, this can be used for building a CL kernel"""
    if is_cl_vector_type(data):
        vector_info = vector_type_lookup(data)
        length = vector_info['length']
        if vector_info['length'] % 2 == 1:
            length += 1
        return vector_info['data_type'] + repr(length)
    return numpy_types_to_cl(data.dtype.type)


def get_correct_cl_data_type_from_data(data):
    """Get the data type for the given data"""
    if is_cl_vector_type(data):
        vector_info = vector_type_lookup(data)
        length = vector_info['length']
        if length % 2 == 1:
            length += 1
        return get_opencl_vector_data_type(length, data_type=vector_info['data_type'])
    return data.dtype.type


def set_correct_cl_data_type(data):
    """Set for the given data the data type given by get_correct_cl_data_type_from_data()"""
    if data is not None:
        if isinstance(data, dict):
            for key, d in data.items():
                data[key] = set_correct_cl_data_type(d)
            return data
        elif isinstance(data, (tuple, list)):
            items = []
            for d in data:
                items.append(set_correct_cl_data_type(d))
            return items
        elif isinstance(data, (numbers.Number,)):
            return np.array(data, dtype=np.float64, order='C')
        else:
            return data.astype(get_correct_cl_data_type_from_data(data))
    return None


def calculate_kernel_range_and_offsets(nmr_problems, items_per_kernel):
    """Calculate a list of ranges and offsets for the given number of problems and the maximum items per kernel.

    Args:
        nmr_problems (int):
            The number of problems we have to run on OpenCL.
        items_per_kernel (int):
            The maximum number of items we want to run per kernel invocation.

    Returns:
        A list with tuples with (range, offset) for each invocation.
    """
    range_offsets = []
    remaining = nmr_problems
    for i in range(int(round(nmr_problems / float(items_per_kernel)))):
        r = min(items_per_kernel, remaining)
        o = nmr_problems - remaining
        remaining -= r
        range_offsets.append((int(r), int(o)))
    return range_offsets


class ParameterCLCodeGenerator(object):

    def __init__(self, device, var_data_dict, prtcl_data_dict, model_data_dict, add_var_data_multipliers=True):
        self._device = device
        self._max_constant_buffer_size = device.get_info(cl.device_info.MAX_CONSTANT_BUFFER_SIZE)
        self._max_constant_args = device.get_info(cl.device_info.MAX_CONSTANT_ARGS)
        self._var_data_dict = var_data_dict
        self._prtcl_data_dict = prtcl_data_dict
        self._model_data_dict = model_data_dict
        self._add_var_data_multipliers = add_var_data_multipliers
        self._kernel_items = self._get_all_kernel_source_items()

    def get_data_struct(self):
        return self._kernel_items['data_struct']

    def get_kernel_param_names(self):
        return self._kernel_items['kernel_param_names']

    def get_data_struct_init_assignment(self, variable_name):
        return 'optimize_data ' + variable_name + ' = {' + ', '.join(self._kernel_items['data_struct_init']) + '};'

    def _get_all_kernel_source_items(self):
        """Get the CL strings for the kernel source items for most common CL kernels in this library."""
        constant_args_counter = 0

        kernel_param_names = []
        data_struct_init = []
        data_struct_names = []

        for key, vdata in self._var_data_dict.items():
            clmemtype = 'global'

            if check_array_fits_cl_memory(vdata, get_correct_cl_data_type_from_data(vdata),
                                          self._max_constant_buffer_size):
                if constant_args_counter < self._max_constant_args:
                    clmemtype = 'constant'
                    constant_args_counter += 1

            param_name = 'var_data_' + key
            cl_data_type = get_cl_data_type_from_data(vdata)

            kernel_param_names.append(clmemtype + ' ' + cl_data_type + '* ' + param_name)
            data_struct_names.append(clmemtype + ' ' + cl_data_type + '* ' + param_name)

            if self._add_var_data_multipliers:
                mult = vdata.shape[1] if len(vdata.shape) > 1 else 1
                data_struct_init.append(param_name + ' + gid * ' + repr(mult))
            else:
                data_struct_init.append(param_name)

        for key, vdata in self._prtcl_data_dict.items():
            clmemtype = 'global'

            if check_array_fits_cl_memory(vdata, get_correct_cl_data_type_from_data(vdata),
                                          self._max_constant_buffer_size):
                if constant_args_counter < self._max_constant_args:
                    clmemtype = 'constant'
                    constant_args_counter += 1

            param_name = 'prtcl_data_' + key
            cl_data_type = get_cl_data_type_from_data(vdata)

            kernel_param_names.append(clmemtype + ' ' + cl_data_type + '* ' + param_name)
            data_struct_init.append(param_name)
            data_struct_names.append(clmemtype + ' ' + cl_data_type + '* ' + param_name)

        for key, vdata in self._model_data_dict.items():
            clmemtype = 'global'
            param_name = 'fixed_data_' + key
            cl_data_type = get_cl_data_type_from_data(vdata)

            data_struct_init.append(param_name)

            if isinstance(vdata, np.ndarray):
                kernel_param_names.append(clmemtype + ' ' + cl_data_type + '* ' + param_name)
                data_struct_names.append(clmemtype + ' ' + cl_data_type + '* ' + param_name)
            else:
                kernel_param_names.append(cl_data_type + ' ' + param_name)
                data_struct_names.append(cl_data_type + ' ' + param_name)

        data_struct = '''
            typedef struct{
                ''' + ('' if data_struct_names else 'constant void* place_holder;') + '''
                ''' + " ".join((name + ";\n" for name in data_struct_names)) + '''
            } optimize_data;
        '''

        return {'kernel_param_names': kernel_param_names,
                'data_struct_names': data_struct_names,
                'data_struct_init': data_struct_init,
                'data_struct': data_struct}


def init_dict_tree():
    return defaultdict(init_dict_tree)