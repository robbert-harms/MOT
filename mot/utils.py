import pyopencl.array as cl_array
import numpy as np
import pyopencl as cl
from functools import reduce
from pkg_resources import resource_filename
import os
from .data_adapters  import DataAdapter


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
    if cl_device_type_str == 'GPU':
        return cl.device_type.GPU
    if cl_device_type_str == 'CPU':
        return cl.device_type.CPU
    if cl_device_type_str == 'ACCELERATOR':
        return cl.device_type.ACCELERATOR
    if cl_device_type_str == 'CUSTOM':
        return cl.device_type.CUSTOM
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
    """Create a dictionary out of the results, which the optimizer can output to the user

    Args:
        results: a 2d (from optimization) or 3d (from sampling) array that needs to be converted to a dictionary.
        param_names (list of str): the names of the parameters, one per column

    Returns:
        dict: the results packed in a dictionary
    """
    results_slice = [slice(None)] * len(results.shape)
    d = {}
    for i in range(len(param_names)):
        results_slice[1] = i
        d[param_names[i]] = results[results_slice]
    return d


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


class TopologicalSort(object):

    def __init__(self, data):
        """Topological sort the the given data. The data should consist of a dictionary structure.

        Args:
            data (dict); dictionary structure where the value is a list of dependencies for that given key.
                as an example ``{'a', (), 'b': ('a',)}``, where ``a`` depends on nothing and ``b`` depends on ``a``.
        """
        self.data = data

    def get_sorted(self):
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

    def get_flattened(self, sort=True):
        """Returns a single flattened list of dependencies.

        Optionally we can sort the list to make the results repeatable.

        Args:
            sort (boolean): if we want to sort the results list

        Returns:
            list: the list of dependencies in constructure order, optionally sorted
        """
        result = []
        for d in self.get_sorted():
            result.extend((sorted if sort else list)(d))
        return result


class ParameterCLCodeGenerator(object):

    def __init__(self, device, var_data_dict, protocol_data_dict, model_data_dict):
        """Generate the CL code for all the parameters in the given dictionaries.

        The dictionaries are supposed to contain DataAdapters.

        Args:
            device: the CL device we want to compile the code for
            var_data_dict (dict[str, CLDataAdapter]): the dictionary with the variable data. That is, the data
                that is different for every problem (but constant over the measurements).
            protocol_data_dict (dict[str, CLDataAdapter]): the dictionary with the protocol data. That is, the data
                that is the same for every problem, but differs per measurement.
            model_data_dict (dict[str, CLDataAdapter]): the dictionary with the model data. That is, the data
                that is the same for every problem and every measurement.
        """
        self._device = device
        self._max_constant_buffer_size = device.get_info(cl.device_info.MAX_CONSTANT_BUFFER_SIZE)
        self._max_constant_args = device.get_info(cl.device_info.MAX_CONSTANT_ARGS)
        self._var_data_dict = var_data_dict
        self._protocol_data_dict = protocol_data_dict
        self._model_data_dict = model_data_dict
        self._kernel_items = self._get_all_kernel_source_items()

    def get_data_struct(self):
        return self._kernel_items['data_struct']

    def get_kernel_param_names(self):
        return self._kernel_items['kernel_param_names']

    def get_data_struct_init_assignment(self, variable_name):
        struct_code = '0'
        if self._kernel_items['data_struct_init']:
            struct_code = ', '.join(self._kernel_items['data_struct_init'])
        return 'optimize_data ' + variable_name + ' = {' + struct_code + '};'

    def _get_all_kernel_source_items(self):
        """Get the CL strings for the kernel source items for most common CL kernels in this library."""
        constant_args_counter = 0

        kernel_param_names = []
        data_struct_init = []
        data_struct_names = []

        for key, data_adapter in self._var_data_dict.items():
            clmemtype = 'global'

            cl_data = data_adapter.get_opencl_data()

            if self._check_array_fits_constant_buffer(cl_data, data_adapter.get_opencl_numpy_type()):
                if constant_args_counter < self._max_constant_args:
                    clmemtype = 'constant'
                    constant_args_counter += 1

            param_name = 'var_data_' + str(key)
            data_type = data_adapter.get_data_type().raw_data_type

            if data_adapter.get_data_type().is_vector_type:
                data_type += data_adapter.get_data_type().vector_length

            kernel_param_names.append(clmemtype + ' ' + data_type + '* ' + param_name)

            mult = cl_data.shape[1] if len(cl_data.shape) > 1 else 1
            if len(cl_data.shape) == 1 or cl_data.shape[1] == 1:
                data_struct_names.append(data_type + ' ' + param_name)
                data_struct_init.append(param_name + '[gid * ' + str(mult) + ']')
            else:
                data_struct_names.append(clmemtype + ' ' + data_type + '* ' + param_name)
                data_struct_init.append(param_name + ' + gid * ' + str(mult))

        for key, data_adapter in self._protocol_data_dict.items():
            clmemtype = 'global'

            cl_data = data_adapter.get_opencl_data()

            if self._check_array_fits_constant_buffer(cl_data, data_adapter.get_opencl_numpy_type()):
                if constant_args_counter < self._max_constant_args:
                    clmemtype = 'constant'
                    constant_args_counter += 1

            param_name = 'protocol_data_' + str(key)
            data_type = data_adapter.get_data_type().raw_data_type

            if data_adapter.get_data_type().is_vector_type:
                data_type += str(data_adapter.get_data_type().vector_length)

            kernel_param_names.append(clmemtype + ' ' + data_type + '* ' + param_name)
            data_struct_init.append(param_name)
            data_struct_names.append(clmemtype + ' ' + data_type + '* ' + param_name)

        for key, data_adapter in self._model_data_dict.items():
            clmemtype = 'global'
            param_name = 'model_data_' + str(key)
            data_type = data_adapter.get_data_type().raw_data_type

            if data_adapter.get_data_type().is_vector_type:
                data_type += data_adapter.get_data_type().vector_length

            data_struct_init.append(param_name)

            if isinstance(data_adapter.get_opencl_data(), np.ndarray):
                kernel_param_names.append(clmemtype + ' ' + data_type + '* ' + param_name)
                data_struct_names.append(clmemtype + ' ' + data_type + '* ' + param_name)
            else:
                kernel_param_names.append(data_type + ' ' + param_name)
                data_struct_names.append(data_type + ' ' + param_name)

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

    def _check_array_fits_constant_buffer(self, array, dtype):
        """Check if the given array when casted to the given type can be fit into the given max_size

        Args:
            array (ndarray): the array we want to fit
            dtype (np data type): the numpy data type we want to use

        Returns:
            boolean: if it fits in the constant memory buffer or not
        """
        return np.product(array.shape) * np.dtype(dtype).itemsize < self._max_constant_buffer_size


def initialize_ranlux(cl_context, nmr_instances, ranluxcl_lux=None, seed=None):
    """Create an opencl buffer with the initialized RanluxCLTab.

    Args:
        cl_environment (CLEnvironment): the environment to use
        cl_context: the context to use (containing the queue and actual context)
        nmr_instances (int): for how many thread instances we should initialize the ranlux cl tab.
        ranluxcl_lux (int): the luxury level of the ranluxcl generator. See the ranluxcl.cl source for details.
        seed (int): the seed to use, see the ranluxcl.cl source for details. If not given (is None) we will use
            the seed number defined in the current configuration.

    Returns:
        cl buffer: the buffer containing the initialized ranlux cl tab for use in the given environment/queue.
    """
    from mot.configuration import get_ranlux_seed

    if seed is None:
        seed = get_ranlux_seed()

    kernel_source = get_ranlux_cl(ranluxcl_lux=ranluxcl_lux)
    kernel_source += '''
        __kernel void init(global ranluxcl_state_t *ranluxcltab){
            ranluxcl_initialization(''' + str(seed) + ''', ranluxcltab);
        }
    '''
    ranluxcltab_buffer = cl.Buffer(cl_context.context,
                                   cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR,
                                   size=np.dtype(cl_array.vec.float4).itemsize * nmr_instances * 7)
    kernel = cl.Program(cl_context.context, kernel_source).build()
    kernel.init(cl_context.queue, (int(nmr_instances), ), None, ranluxcltab_buffer).wait()
    return ranluxcltab_buffer


def is_scalar(value):
    """Test if the given value is a scalar.

    This function also works with memory mapped array values, in contrast to the numpy is_scalar method.

    Args:
        value: the value to test for being a scalar value

    Returns:
        boolean: if the given value is a scalar or not
    """
    return np.isscalar(value) or (isinstance(value, np.ndarray) and (len(np.squeeze(value).shape) == 0))


def get_ranlux_cl(ranluxcl_lux=None):
    """Get the code for the RanLux generator.

    Args:
        ranluxcl_lux (int): the luxury level of the ranluxcl generator. See the ranluxcl.cl source for details.

    Returns:
        str: the CL code string for the complete RanLux RNG.
    """
    from mot.configuration import get_ranlux_lux_factor

    if ranluxcl_lux is None:
        ranluxcl_lux = get_ranlux_lux_factor()

    cl_source = '#define RANLUXCL_LUX ' + str(ranluxcl_lux) + "\n"
    cl_source += open(os.path.abspath(resource_filename('mot', 'data/opencl/ranluxcl.h'),), 'r').read()
    cl_source += open(os.path.abspath(resource_filename('mot', 'data/opencl/ranluxcl.cl'), ), 'r').read()

    return cl_source
